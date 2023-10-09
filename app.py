import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import torch
import calendar as cl
from pred_in import prediction_input

app = Flask(__name__)
# model = pickle.load(open('model.pkl', 'rb'))
model = torch.jit.load('model_scripted.pt', map_location= torch.device("cpu"))
model.eval()
scaler = pickle.load(open('scalerx.pkl', 'rb'))
scaler1 = pickle.load(open('scalery.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [x.lower() for x in request.form.values()]
    day_dict = prediction_input()    
    sum = 0
    for j in day_dict[int_features[0]]:
        final_features = torch.tensor(scaler.transform(np.array(j).reshape(1,-1)), dtype=torch.float32).to(device='cpu')
        b = model(final_features).detach().numpy()
        prediction = scaler1.inverse_transform(b.reshape(-1,1))
        sum += int(prediction[0])

    return render_template('index.html', prediction_text='No. of receipts will be {}'.format(sum))

@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    # prediction = model.predict([np.array(list(data.values()))])

    # int_features = [int(x) for x in request.form.values()]
    final_features = torch.tensor(scaler.transform(torch.tensor(data).reshape(-1,1)), dtype=torch.float32).to(device='cpu')

    b = model(final_features).detach().numpy()
    prediction = scaler1.inverse_transform(b.reshape(-1,1))

    output = int(prediction[0])

    # output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)