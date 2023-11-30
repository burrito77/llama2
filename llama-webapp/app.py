from flask import Flask, request, jsonify, render_template

from llama import Llama

app = Flask(__name__)

# Initialize the Llama generator
ckpt_dir = "llama_checkpoint/"
tokenizer_path = "tokenizers/"

generator = Llama.build(
    ckpt_dir=ckpt_dir,
    tokenizer_path=tokenizer_path,
    max_seq_len=512,
    max_batch_size=8,
)

@app.route('/generate', methods=['POST', 'GET'])
def generate_response():
    if request.method == 'POST':
        data = request.get_json()
        instructions = data.get('instructions', [])
        
        # Generate responses using the Llama generator
        results = generator.chat_completion(
            instructions,
            max_gen_len=None,  # Set to your desired value
            temperature=0.2,  # Set to your desired value
            top_p=0.95,  # Set to your desired value
        )
        
        # Extract and format the generated responses
        formatted_responses = []
        for instruction, result in zip(instructions, results):
            user_message = [msg['content'] for msg in instruction if msg['role'] == 'user']
            system_message = [msg['content'] for msg in instruction if msg['role'] == 'system']
        
            formatted_responses.append({
                'user_message': user_message,
                'system_message': system_message,
                'generated_response': result['generation']['content']
            })
        
        return jsonify({"responses": formatted_responses})
    
    # If the request method is GET, render an HTML form
    return render_template('form.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
