from langchain.memory import ConversationBufferMemory
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# Initialize a simple conversation memory
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Example of adding a message to memory
memory.chat_memory.add_user_message("Hello!")
memory.chat_memory.add_ai_message("Hi there! How can I help you today?")

# Retrieve the chat history
chat_history = memory.load_memory_variables({})
print("Chat History:", chat_history)

# Initialize the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    device_map="mps",
    torch_dtype="auto", 
    trust_remote_code=True,
    attn_implementation="eager"
)

tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")

# Create generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer
)

# Generation settings
generation_args = {
    "max_new_tokens": 500,
    "return_full_text": False,
    "temperature": 0.5,
    "do_sample": False,
    "use_cache": False
}

# Function to generate response
def get_model_response(user_input):
    messages = [{"role": "user", "content": user_input}]
    response = pipe(messages, **generation_args)
    return response[0]['generated_text']

# Example usage with memory
user_message = "Tell me a joke"
memory.chat_memory.add_user_message(user_message)

model_response = get_model_response(user_message)
memory.chat_memory.add_ai_message(model_response)

# Get updated chat history
chat_history = memory.load_memory_variables({})
print("\nUpdated Chat History:", chat_history)
