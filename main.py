from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import base64
from fastapi.middleware.cors import CORSMiddleware
import requests
import pathlib
import os
# from PIL import Image
import io
from pathlib import Path
import google.generativeai as genai
from dotenv import load_dotenv,dotenv_values
# from deepface import DeepFace

load_dotenv()

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load YOLOv5 model

# weight = 'plant.pt'
# absolute_path = os.path.abspath(weight)
# device = select_device('')
# model = attempt_load(absolute_path, device)
# stride = int(model.stride.max())


app = FastAPI()


class ImageData(BaseModel):
    imagedata: str

class Input(BaseModel):
    image_url: str
    profile_pic_url:str

origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"Plant": "ASAP"}

# Load the pre-trained Haar Cascade face detection model
# haar = 'C:\D\KODING SHITZ\personal\Diversion\Backend\haarcascade_frontalface_alt2.xml'
# face_cascade = cv2.CascadeClassifier(haar)

@app.post("/get_image_base64")
def get_image_base64(image_url: str):
    try:
        # Fetch the image
        print(image_url)
        response = requests.get(image_url)
        response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)

        # Convert the image to base64
        base64_data = base64.b64encode(response.content).decode('utf-8')
        return base64_data
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching the image: {str(e)}")

@app.post("/detect_face")
async def detect_face(imagedata:str):
    # Read the image file
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        shape = img.shape
        x,y = shape[0]//2,shape[1]//2
        # print(x,y)
        # Perform object detection
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (x,y))

        # Convert the image to grayscale for face detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.01, minNeighbors=5)

        # Draw rectangles around the detected faces
        for (x, y, w, h) in faces:
            # cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            img_cropped = img[y:y+h,x:x+w]
            ### cv2.imwrite("./face.jpg", img_cropped)
        # Save the result image (optional)
        print(faces)
        if not len(faces):
            return False
        else:
            return True


        # Convert the result image to bytes to be sent in the response
        # _, img_encoded = cv2.imencode('.jpg', img)
        # img_bytes = img_encoded.tobytes()

        # # Return the result as JSON response
        # return JSONResponse(content={"message": "Faces detected successfully", "image": img_bytes.decode("utf-8")})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

def base64_to_image(base64_string):
    image_bytes = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_bytes))
    return image

@app.post("/detect_plant")
def detect_plant(imagedata:str):
    try:
        # Decode base64 string and read it as an image
        image_bytes = base64.b64decode(imagedata)
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Perform object detection
        img = cv2.resize(img, (640, 640))
        image = img
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        img = img.permute(2, 0, 1).unsqueeze(0)

        pred = model(img)[0]
        pred = non_max_suppression(pred, 0.2, 0.5)

        plant_bool = False
        for det in pred[0]:
            x1, y1, x2, y2, conf, cls = det
            x,y,w,h = (int(x1.numpy()), int(y1.numpy()), int(x2.numpy()), int(y2.numpy()))
            # print(coordinates)
            # Display the coordinates
            cls = int(cls.item())
            print("cls",cls)
            if cls == 0:
                plant = 'plant detected'
                plant_bool = True
                image_cropped = image[x:x+w,y:y+h]
                ### cv2.imwrite("plant.jpg", image_cropped)

        if plant_bool:
            output = plant
            print(output)
            return plant_bool
        return plant_bool
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare_faces")
def compare_faces(image:str,profile_pic:str):
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    # Set up the model
    generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]

    model = genai.GenerativeModel(model_name="gemini-pro-vision",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
        # Validate that an image is present

    image_parts = [
    {
        "mime_type": "image/jpeg",
        "data": image
    },
    {
        "mime_type": "image/jpeg",
        "data": profile_pic
    },
    ]

    prompt_parts = [
    image_parts[0],
    "\n\n",
    image_parts[1],
    "\n\nAnalyze the faces in the base64 images given. If both the images have the same face just return 'True' else just return 'False'",
    ]

    response = model.generate_content(prompt_parts)
    if response.text[1:]=="True":
        return True
    else:
        return False 
    
@app.post("/find_plants")
def find_plants(image:str):
    api_key = os.getenv("GEMINI_API_KEY")
    genai.configure(api_key=api_key)

    # Set up the model
    generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]

    model = genai.GenerativeModel(model_name="gemini-pro-vision",
                                generation_config=generation_config,
                                safety_settings=safety_settings)
        # Validate that an image is present

    image_parts = [
    {
        "mime_type": "image/jpeg",
        "data": image
    },
    ]

    prompt_parts = [
    image_parts[0],
    "\n\n",
    "\n\nAnalyze the given base64 image. if there is freshly planted baby plant present in the picture then return 'True' else return 'False'",
    ]

    response = model.generate_content(prompt_parts)
    if response.text[1:]=="True":
        return True
    else:
        return False 

@app.post("/verify")
def verify(input : Input):
    try:
        img1 = get_image_base64(input.image_url)
        img2 = get_image_base64(input.profile_pic_url)
        if find_plants(img1) and compare_faces(img1,img2):
            return True
        else:
            return False
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error fetching the image: {str(e)}")
    


@app.post("/generative_ai")
async def generative_ai(prompt:str):


    API_BASE_URL = "https://api.cloudflare.com/client/v4/accounts/83268b3dbc596d0ff47c79398333a126/ai/run/"
    API_KEY = os.getenv("CLOUDFLARE_API_KEY")
    headers = {"Authorization": f"Bearer {API_KEY}"}
    inputs = {
        "messages":[
        { "role": "system", "content": "You are a friendly assistant with knowledge about botany and flora that answers user's queries in an easy to understand manner" },
        { "role": "user", "content": f"{prompt}"}
    ]
    }
    response = requests.post(f"{API_BASE_URL}@cf/meta/llama-2-7b-chat-int8", headers=headers, json=inputs)
    return response.json()["result"]["response"]
    


