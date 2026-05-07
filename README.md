# Motor Leakage Detection System using Machine Learning

A real-time motor leakage detection system powered by a neural network. This project uses PyTorch to train a classifier that identifies air intake and exhaust leaks in turbocharged diesel engines based on sensor readings.

## Features

- **Neural Network Classifier**: Deep learning model trained on synthetic engine data to detect leaks in 5 zones (NONE, A, B, C, D)
- **Real-time Web Interface**: Interactive Flask web app with live sensor input sliders
- **Visual Diagnostics**: 
  - Animated neural network signal propagation visualization
  - Engine block diagram with leak zone highlighting
  - Confidence probability bars for each classification
- **Synthetic Data Generation**: Creates realistic training data based on engine physics
- **Model Training**: Automated training pipeline with validation and model saving

## Leak Zones Detected

- **NONE**: System healthy - no leak detected
- **Zone A**: Airflow drop (Airflow meter → Turbo inlet)
- **Zone B**: Intake pressure drop + EGT rise (Compressor → Intercooler)
- **Zone C**: EGT rise only (Exhaust Manifold → Turbine)
- **Zone D**: Exhaust pressure drop (Test cell ducting)

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/MotorLeakageFinder_usingML.git
cd MotorLeakageFinder_usingML
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

Run the training script to train the neural network:
```bash
python ML_prog.py
```

This will:
- Generate synthetic training data if not present
- Train the model for 100 epochs
- Save the trained model as `leaknet.pth`

### Running the Web App

Start the Flask web server:
```bash
python app.py
```

Open your browser and navigate to `http://localhost:5000`

### Using the Interface

1. Adjust the sensor reading sliders (RPM, Airflow, EGT, Intake Pressure, Exhaust Pressure)
2. Click "ANALYZE" to get real-time predictions
3. View the neural network animation, engine diagram highlighting, and confidence scores

## Project Structure

```
├── app.py                 # Flask web application
├── ML_prog.py            # Model training and inference script
├── data_creator.py       # Synthetic data generation
├── leaknet.pth           # Trained PyTorch model (generated)
├── healthy_train.csv     # Training data for healthy engines
├── leak_test.csv         # Training data for leaky engines
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html        # Web interface
└── README.md            # This file
```

## Model Architecture

The neural network consists of:
- Input layer: 5 features (RPM, AF, EGT, IP, EP)
- Hidden layer 1: 64 neurons with BatchNorm, ReLU, Dropout
- Hidden layer 2: 32 neurons with BatchNorm, ReLU, Dropout
- Output layer: 5 classes (leak zones)

## Data Generation

The synthetic data is created based on real engine physics:
- Healthy data: Normal operating ranges with realistic correlations
- Leak data: Perturbed sensor readings specific to each leak zone
- Features are standardized using StandardScaler before training

## Dependencies

- PyTorch - Deep learning framework
- Flask - Web framework
- NumPy - Numerical computing
- Pandas - Data manipulation
- Scikit-learn - Machine learning utilities

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source. Feel free to use and modify as needed.

## Acknowledgments

- Built with PyTorch for neural network implementation
- Flask for the web interface
- Inspired by real-world engine diagnostic systems</content>
<parameter name="filePath">c:\Users\tamil\Desktop\_\projects\MotorLeakageFinder_usingML\README.md