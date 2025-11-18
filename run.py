from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load your trained model and tokenizer
MODEL_DIR = "cli_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

def nl_to_cli(nl_command: str, max_tokens: int = 128) -> str:
    """
    Convert a natural language command to CLI commands using the trained model.
    
    Args:
        nl_command (str): Natural language input.
        max_tokens (int): Maximum number of tokens to generate.
    
    Returns:
        str: Generated CLI commands.
    """
    inputs = tokenizer(nl_command, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    cli_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return cli_output

if __name__ == "__main__":
    print("NL → CLI Interactive")
    print("Type 'exit' to quit.\n")
    
    while True:
        user_input = input("NL> ")
        if user_input.lower() in ("exit", "quit"):
            break
        cli_command = nl_to_cli(user_input)
        print(f"CLI> {cli_command}\n")
