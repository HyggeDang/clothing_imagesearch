from flask import Flask, render_template, request, jsonify
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    image_file = request.files['image']
    image_path = os.path.join('uploads', image_file.filename)
    image_file.save(image_path)
    
    similar_images = find_similar_images(image_path)
    
    results = []
    for img_path, dist in similar_images:
        results.append({'image': img_path, 'distance': dist})
    
    return jsonify({'results': results})

if __name__ == '__main__':
    app.run(debug=True)
