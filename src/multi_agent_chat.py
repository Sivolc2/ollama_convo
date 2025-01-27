import ollama
import sys
import yaml
from typing import Dict, Any, List
import re
import asyncio
from concurrent.futures import ThreadPoolExecutor

class Agent:
    def __init__(self, name: str, model_name: str, system_prompt: str):
        self.name = name
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.conversation_history: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the agent's conversation history."""
        self.conversation_history.append({
            'role': role,
            'content': content
        })
    
    def get_clean_response(self, response: str) -> str:
        """Remove thinking process from response."""
        # Remove any text between <think> tags
        clean_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return clean_response.strip()

    def stream_chat(self, message: str, chat_settings: Dict[str, Any]) -> None:
        """Stream a chat response from the agent."""
        try:
            # Add user message to history
            self.add_message('user', message)
            
            # Print agent name at the start
            print(f"\n{self.name}: ", end='', flush=True)
            
            # Get streaming response from model
            full_response = ""
            stream = ollama.chat(
                model=self.model_name,
                messages=self.conversation_history,
                stream=True,
                options={
                    "temperature": chat_settings.get('temperature', 0.7)
                }
            )
            
            for chunk in stream:
                content = chunk['message']['content']
                # Accumulate the full response
                full_response += content
                # Print the chunk
                print(content, end='', flush=True)
            
            # Clean the full response and add to history
            clean_response = self.get_clean_response(full_response)
            self.add_message('assistant', clean_response)
            print()  # New line after response
            
        except Exception as e:
            print(f"Error from {self.name}: {str(e)}")

def stream_agent_response(agent: Agent, message: str, chat_settings: Dict[str, Any]) -> None:
    """Helper function to stream an agent's response in a separate thread."""
    agent.stream_chat(message, chat_settings)

class MultiAgentChat:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.agents: Dict[str, Agent] = {}
        self.setup_agents()
        self.executor = ThreadPoolExecutor(max_workers=10)  # Limit concurrent agents
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return {}
    
    def setup_agents(self) -> None:
        """Create agents from configuration."""
        for model_key, model_config in self.config['models'].items():
            # Create the custom model first
            try:
                ollama.create(
                    model=model_config['custom_name'],
                    from_=model_config['base_model'],
                    system=model_config['system_prompt']
                )
                
                # Create the agent
                agent = Agent(
                    name=model_key,
                    model_name=model_config['custom_name'],
                    system_prompt=model_config['system_prompt']
                )
                self.agents[model_key] = agent
                print(f"Created agent: {model_key}")
                
            except Exception as e:
                print(f"Error creating agent {model_key}: {str(e)}")
    
    def list_agents(self) -> None:
        """List all available agents."""
        print("\nAvailable agents:")
        for agent_name in self.agents.keys():
            print(f"- {agent_name}")
    
    def chat_with_agents(self, message: str, agent_names: List[str]) -> None:
        """Get streaming responses from specified agents concurrently."""
        chat_settings = self.config.get('chat_settings', {})
        
        print("\nResponses:")
        futures = []
        
        # Start streaming for each agent concurrently
        for agent_name in agent_names:
            if agent_name in self.agents:
                agent = self.agents[agent_name]
                future = self.executor.submit(stream_agent_response, agent, message, chat_settings)
                futures.append(future)
            else:
                print(f"\n{agent_name}: Agent not found")
        
        # Wait for all responses to complete
        for future in futures:
            future.result()

def main():
    chat_system = MultiAgentChat()
    
    if not chat_system.agents:
        print("No agents available. Exiting.")
        sys.exit(1)
    
    chat_system.list_agents()
    
    # Get initial agent selection
    print("\nEnter agent names separated by commas (or 'all' for all agents)")
    agent_input = input("Choose agents: ").strip()
    
    if agent_input.lower() == 'all':
        selected_agents = list(chat_system.agents.keys())
    else:
        selected_agents = [name.strip() for name in agent_input.split(',')]
    
    # Start chat loop
    while True:
        user_input = input("\nEnter your message (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
        
        chat_system.chat_with_agents(user_input, selected_agents)

if __name__ == "__main__":
    main() 