import ollama
import sys
import yaml
from typing import Dict, Any

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the configuration file
    
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config: {str(e)}")
        return {}

def create_custom_model(config: Dict[str, Any], model_key: str) -> str:
    """
    Create a custom model with a specific system prompt from configuration.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        model_key (str): Key of the model in configuration
    
    Returns:
        str: Name of the created model
    """
    try:
        model_config = config['models'][model_key]
        base_model = model_config['base_model']
        custom_name = model_config['custom_name']
        system_prompt = model_config['system_prompt']

        print(f"Creating custom model '{custom_name}' from {base_model}...")
        ollama.create(
            model=custom_name,
            from_=base_model,
            system=system_prompt
        )
        print(f"Custom model '{custom_name}' created successfully!")
        return custom_name
    except Exception as e:
        print(f"Error creating custom model: {str(e)}")
        return base_model

def chat_with_model(model_name: str, message: str, chat_settings: Dict[str, Any]) -> None:
    """
    Chat with a locally hosted Ollama model with streaming output.
    
    Args:
        model_name (str): Name of the model to use
        message (str): Message to send to the model
        chat_settings (Dict[str, Any]): Chat configuration settings
    """
    try:
        # Create a streaming conversation with the model
        stream = ollama.chat(
            model=model_name,
            messages=[{
                'role': 'user',
                'content': message
            }],
            stream=chat_settings.get('stream', True),
            options={
                "temperature": chat_settings.get('temperature', 0.7)
            }
        )
        
        # Print the response as it comes in
        print("\nModel response:", flush=True)
        for chunk in stream:
            content = chunk['message']['content']
            print(content, end='', flush=True)
        print("\n")  # Add a newline after response is complete
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def list_available_personas(config: Dict[str, Any]) -> None:
    """
    List all available persona configurations.
    """
    print("\nAvailable personas:")
    for model_key in config['models'].keys():
        print(f"- {model_key}")

if __name__ == "__main__":
    print("Starting chat with Ollama model...")
    
    # Load configuration
    config = load_config()
    if not config:
        print("Failed to load configuration. Exiting.")
        sys.exit(1)
    
    # List available personas
    list_available_personas(config)
    
    # Get model choice from user or use default
    default_model = config.get('default_model', 'mario')
    model_choice = input(f"\nChoose a persona (press Enter for default '{default_model}'): ").strip()
    if not model_choice:
        model_choice = default_model
    
    if model_choice not in config['models']:
        print(f"Invalid persona choice. Using default '{default_model}'")
        model_choice = default_model
    
    # Create the chosen model
    custom_model = create_custom_model(config, model_choice)
    chat_settings = config.get('chat_settings', {})
    
    # Start chat loop
    while True:
        user_input = input("\nEnter your message (or 'quit' to exit): ")
        if user_input.lower() == 'quit':
            break
            
        chat_with_model(
            model_name=custom_model,
            message=user_input,
            chat_settings=chat_settings
        ) 