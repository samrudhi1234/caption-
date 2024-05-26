#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Define the Streamlit app code as a string
streamlit_app_code = """
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import nltk
from nltk.corpus import stopwords

# Download the stopwords
nltk.download('stopwords')

# Initialize the caption model and processor
caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

stop_words = set(stopwords.words('english'))

def generate_hashtags(caption):
    # Extract keywords or phrases from the caption
    words = caption.split()
    keywords = [word.lower() for word in words if word.lower() not in stop_words]

    # Limit to a certain number of hashtags
    keywords = keywords[:4]  # Example: Take first 4 keywords

    # Generate hashtags from keywords
    hashtags = ['#' + word for word in keywords]

    return ' '.join(hashtags)  # Concatenate hashtags into a string

def enhance_caption(caption, hashtags):
    # Add predefined text to make the caption more engaging for Instagram
    additional_text = "\\n\\nWhat do you think? 📸✨"
    common_hashtags = "#photography #instadaily #photooftheday #instagood #nature"

    # Create the enhanced caption
    enhanced_caption = f"{caption}{additional_text}\\n\\n{hashtags} {common_hashtags}"

    return enhanced_caption

def main():
    st.title("Caption AI App")

    uploaded_file = st.file_uploader("Choose an image...", type="jpg")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        st.write("")
        st.write("Generating caption...")

        inputs = processor(images=image, return_tensors="pt")
        out = caption_model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        hashtags = generate_hashtags(caption)
        enhanced_caption = enhance_caption(caption, hashtags)

        st.write("Caption:")
        st.write(enhanced_caption)

if __name__ == "__main__":
    main()
"""

# Save the string to a file
with open("app.py", "w") as file:
    file.write(streamlit_app_code)

print("app.py has been created successfully.")

