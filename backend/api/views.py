from rest_framework.response import Response
from rest_framework.decorators import api_view
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --- Simple test route ---
@api_view(['GET'])
def test_api(request):
    return Response({"message": "Neural Network Optimization Backend is Running!"})

# --- Neural Network prediction route ---
@api_view(['GET'])
def predict_study_period(request):
    # Example: simple neural network (we'll replace later with real trained model)
    model = Sequential([
        Dense(8, activation='relu', input_shape=(3,)),
        Dense(4, activation='relu'),
        Dense(1, activation='linear')
    ])
    
    model.compile(optimizer='adam', loss='mse')

    # Example dummy input
    X_sample = np.array([[3, 5, 7]])  # e.g. [hours_studied, attendance, prev_score]
    y_pred = model.predict(X_sample,  verbose=0)

    return Response({"predicted_study_period": float(y_pred[0][0])})
