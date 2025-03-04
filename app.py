import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Define the path to your dataset
dataset_path = "C:/Users/Anupriya V M/Downloads/bird species/train"  # Update this with the actual path to your dataset folder

# Load the pre-trained model (ensure this happens once and before any prediction)
@st.cache_resource
def load_bird_model():
    try:
        model = load_model('bird_species_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model at the start
model = load_bird_model()

# Define image size
IMG_SIZE = (224, 224)

# Get the class labels from the folder names
class_labels = sorted(os.listdir(dataset_path))  # This gets the folder names and sorts them alphabetically

# Custom user-agent
USER_AGENT = 'MyBirdApp/1.0 (https://example.com; contact@example.com)'

# Function to get bird description from Wikipedia using requests
def get_wikipedia_description(bird_species):
    bird_species_formatted = bird_species.lower().replace(' ', '_')
    url = f"https://en.wikipedia.org/wiki/{bird_species_formatted}"
    headers = {'User-Agent': USER_AGENT}
    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        if paragraphs:
            description = []
            for para in paragraphs:
                text = para.get_text().strip()
                if text:
                    description.append(text)
                if len(description) >= 3:  # Collecting up to 3 paragraphs
                    break
            return "\n\n".join(description)  # Return the collected paragraphs joined by newlines
        return "No description available."
    else:
        return "Description not available on Wikipedia."

# Function to predict bird species
def predict_bird_species(image_file):
    if model is None:
        st.error("Model could not be loaded. Please check the model file.")
        return None

    # Load and preprocess the image
    img = load_img(image_file, target_size=IMG_SIZE)
    img_array = img_to_array(img) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Get predictions
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions, axis=1)  # Get the index of the predicted class

    # Fetch the corresponding bird species name using folder names
    bird_species = class_labels[predicted_class_idx[0]]

    return bird_species

# Function to get bird occurrence data (replace with actual data fetching)
def get_bird_occurrence_data(bird_species):
    # Example: You should query a real database or API for actual occurrences
    # Below is dummy data for demonstration
    occurrence_data = pd.DataFrame({
        'lat': [22.3511, 19.0760, 28.7041, 51.5074, -34.6037],  # Replace with actual coordinates
        'lon': [78.6677, 72.8777, 77.1025, -0.1278, -58.3816],
        'count': [50, 30, 40, 20, 60]  # Replace with actual observation counts
    })

    # Example filtering if you have bird-specific data
    return occurrence_data

# Function to plot heatmap and scatter plot based on bird species occurrence data
def plot_species_map(bird_species):
    # Get the occurrence data for the predicted bird species
    bird_occurrences = get_bird_occurrence_data(bird_species)

    if bird_occurrences.empty:
        st.write("No occurrence data available for this species.")
        return

    # Create a Folium map centered on the world
    world_map = folium.Map(location=[20, 0], zoom_start=2)

    # Add a heatmap layer using the bird occurrence data
    heat_data = [[row['lat'], row['lon']] for index, row in bird_occurrences.iterrows()]
    HeatMap(heat_data).add_to(world_map)

    # Add scatter points with observation counts
    for i, row in bird_occurrences.iterrows():
        folium.CircleMarker(
            location=(row['lat'], row['lon']),
            radius=5,
            weight=1,
            color="blue",
            fill=True,
            fill_color="blue",
            fill_opacity=0.6,
            popup=f"Observations: {row['count']}"
        ).add_to(world_map)

    # Display the map in Streamlit
    st_folium(world_map, width=700, height=500)

# Streamlit interface for predictions
st.title('Bird Species Classification and Occurrence Map')
st.write("Upload an image of a bird, and the model will predict its species and show occurrence data.")

# Upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Make prediction
    bird_species = predict_bird_species(uploaded_file)

    if bird_species:
        # Debugging output for the predicted species
        st.write(f"Predicted species: **{bird_species}**")

        # Fetch and display the bird description from Wikipedia
        description = get_wikipedia_description(bird_species)
        st.write(f"Description from Wikipedia: **{description}**")

        # Plot the occurrence map and scatter plot for the predicted species
        st.write("Occurrence data globally for the predicted species:")
        plot_species_map(bird_species)
