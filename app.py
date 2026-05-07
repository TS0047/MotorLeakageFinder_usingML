from flask import Flask, render_template, request, jsonify
import torch, torch.nn as nn, numpy as np

app = Flask(__name__)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeakNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5,64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32,5)
        )
    def forward(self, x): return self.net(x)

ckpt  = torch.load("leaknet.pth", map_location=DEVICE, weights_only=False)
model = LeakNet().to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()
INV_MAP = {0:"NONE",1:"A",2:"B",3:"C",4:"D"}

@app.route('/')
def index(): return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    d = request.json
    x = np.array([[d['RPM'],d['AF'],d['EGT'],d['IP'],d['EP']]], dtype=np.float32)
    x = (x - ckpt['scaler_mean']) / ckpt['scaler_scale']
    with torch.no_grad():
        out   = model(torch.tensor(x, dtype=torch.float32).to(DEVICE))
        probs = torch.softmax(out, dim=1).cpu().numpy()[0].tolist()
    return jsonify({'zone': INV_MAP[out.argmax(1).item()], 'probs': probs})

if __name__ == '__main__':
    print(f"Device: {DEVICE}  →  http://localhost:5000")
    app.run(debug=False, port=5000)