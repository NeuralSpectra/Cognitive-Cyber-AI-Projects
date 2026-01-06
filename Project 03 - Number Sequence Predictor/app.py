import streamlit as st
import tensorflow as tf
import numpy as np
import warnings

warnings.simplefilter("ignore")

model = tf.keras.models.load_model('Model-Number-Sequence.h5')

st.title('🔢 Number Sequence Predictor')
st.markdown("""
Enter a **sequence of numbers (comma-separated)** to predict the next value.  
**Examples:**
- `0, 10, 20, 30, 40, 50, 60`  
- `0, 5, 10, 15, 20, 25, 30`  
- `0, 20, 40, 60, 80, 100, 120`  
*(Note: Please enter **up to 7 numbers**)*
""")

user_input = st.text_input('✏️ Enter Your Sequence:')

if user_input:
    try:
        input_sequence = [float(x.strip()) for x in user_input.split(',') if x.strip() != '']

        if len(input_sequence) < 2:
            st.error("⚠️ Please Enter At Least 2 Numbers To Form A Sequence.")
        elif len(input_sequence) > 7:
            st.error("⚠️ You Can Only Enter Up To 7 Numbers.")
        else:
            n_steps = 7
            n_features = 1

            if len(input_sequence) < n_steps:
                pad_length = n_steps - len(input_sequence)
                input_sequence = [0]*pad_length + input_sequence

            input_array = np.array(input_sequence).reshape(1, n_steps, n_features)

            next_number = model.predict(input_array)
            predicted_value = next_number[0][0]

            st.success(f"🎯 **Predicted Next Number:** {predicted_value:.2f}")

    except ValueError:
        st.error("❌ Invalid Input! Please Enter Only Numeric Values Separated By Commas.")


