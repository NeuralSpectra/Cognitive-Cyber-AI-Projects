import streamlit as st
from PIL import Image
import pytesseract
import warnings

warnings.simplefilter("ignore")

pytesseract.pytesseract.tesseract_cmd = r'C:\Users\HP\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'

def extract_text(image):

    try:
        text = pytesseract.image_to_string(image)
        return text
    
    except Exception as e:
        return f"Error Occurred During OCR Processing: {e}"

def main():
    st.title("Extract Text From Image")
    
    uploaded_file = st.file_uploader("Choose An Image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:

        image = Image.open(uploaded_file)
        
        st.image(image, caption='Uploaded Image.', use_container_width=True)
        
        if st.button('Extract Text'):

            extracted_text = extract_text(image)
            
            st.text_area("Extracted Text", extracted_text, height=300)
        else:
            st.write("Click 'Extract Text' To Extract Text From The Image.")
    else:
        st.write("Upload An Image To Get Started.")
    
if __name__ == "__main__":
    main()

