import joblib
import pickle
import warnings

warnings.simplefilter("ignore")

with open('Finalized-Model.pickle', 'rb') as f:
    model = pickle.load(f)

joblib.dump(model, 'Finalized-Model-Compressed.pkl', compress=9)

print("Compressed Model Saved As 'Finalized-Model-Compressed.pkl'")

