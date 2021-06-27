import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import smtplib
import pywhatkit
import datetime as dt
import json
import subprocess
import threading
import time

# Load HAAR face classifier
face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



# Load functions

#Face extractor function
def face_extractor(img):
    # Function detects faces and returns the cropped face
    # If no face detected, it returns the input image
    
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)    
    if faces == ():
        return None
    
    # Crop all faces found
    for (x,y,w,h) in faces:
        cropped_face = img[y:y+h, x:x+w]
    return cropped_face

#Model_training function
def training_model():
    # Initialize Webcam
    cap = cv2.VideoCapture(0)
    count = 0

    # Collect 100 samples of your face from webcam input
    while True:

        ret, frame = cap.read()
        if face_extractor(frame) is not None:
            count += 1
            face = cv2.resize(face_extractor(frame), (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Save file in specified directory with unique name
            file_name_path = './faces/user1/' + str(count) + '.jpg'
            cv2.imwrite(file_name_path, face)

            # Put count on images and display live count
            cv2.putText(face, str(count), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            cv2.imshow('Face Cropper', face)


        if cv2.waitKey(1) == 13 or count == 100: #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()      
    print("Collecting Samples Complete")

    # Get the training data we previously made
    data_path = './faces/user1/'
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]

    # Create arrays for training data and labels
    Training_Data, Labels = [], []

    # Open training images in our datapath
    # Create a numpy array for training data
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)

    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)

    # Initialize facial recognizer
    # model = cv2.face.createLBPHFaceRecognizer()
    # NOTE: For OpenCV 3.0 use cv2.face.createLBPHFaceRecognizer()
    # pip install opencv-contrib-python
    # model = cv2.createLBPHFaceRecognizer()

    model  = cv2.face_LBPHFaceRecognizer.create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Model trained successfully")
    return model
    
    
#face detector function
def face_detector(img, size=0.5):
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return img, []
    
    
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2)
        roi = img[y:y+h, x:x+w]
        roi = cv2.resize(roi, (200, 200))
    return img, roi



#mail function
def mail(body, to, from):    
    smtp = smtplib.SMTP('smtp.gmail.com:587')
    smtp.starttls()
    smtp.login("YOUR_EMAIL_ID","Token from Google")
    from_addr=from
    to_addr=to
    smtp.sendmail(from_addr, to_addr, body)
    
    
#Whatsapp function    
def whatsapp(text, phone):   
    x=dt.datetime.now()
    hrs=int(x.strftime("%H"))
    mins=int(x.strftime("%M"))
    pywhatkit.sendwhatmsg(f'+91{phone}', text, hrs ,mins+2, 40, print_wait_time=True, tab_close=True)    
    
#AWS function
def aws_func():
    command_list=['''aws ec2 run-instances --block-device-mappings "DeviceName='/dev/xvdh', Ebs={VolumeSize=10, VolumeType="gp2"}" --image-id ami-0a9d27a9f4f5c0efc --instance-type t2.micro --subnet-id subnet-a5dfa0e9 --security-group-ids sg-085c4d7cb316efb29 --count 1 --key-name aws_key --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=Face_Detection_Instance}]''',
    '''aws ec2 create-volume --availability-zone "ap-south-1b" --size 5''']
    try:
        for command in command_list:
            result=subprocess.run(command, capture_output=True)
            if result.stderr:
                raise subprocess.CalledProcessError(
                        returncode = result.returncode,
                        cmd = result.args,
                        stderr = result.stderr
                        )
            if result.stdout:
                print(result.stdout)

    except subprocess.CalledProcessError as error:
        if "not recognized" in error.stderr.decode("utf-8"):
                print ('wrongcommand does not exist')

    describe_instances=json.loads(subprocess.getoutput("aws ec2 describe-instances"))
    describe_volumes=json.loads(subprocess.getoutput("aws ec2 describe-volumes"))
    for i in describe_instances["Reservations"][0]["Instances"]:
        if i["Tags"][0]["Value"] == "Face_Detection_Instance":
            instance_id=i["InstanceId"]


    for i in describe_volumes["Volumes"]:
            if i["Attachments"] is []:
                volume_id=i["VolumeId"]
                print(json.loads(subprocess.getoutput("aws ec2 attach-volume --device /dev/sdh --instance-id {} --volume-id {}".format(instance_id, volume_id))))
                print("Attached")


def face_recognition(model, user):
    cap = cv2.VideoCapture(0)
    while True:

        ret, frame = cap.read()

        image, face = face_detector(frame)
        try:
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            # Pass face to prediction model
            # "results" comprises of a tuple containing the label and the confidence value
            results = model.predict(face)

            if results[1] < 500:
                confidence = int( 100 * (1 - (results[1])/400) )
                display_string = str(confidence) + '% Confident it is User'

            cv2.putText(image, display_string, (100, 120), cv2.FONT_HERSHEY_COMPLEX, 1, (255,120,150), 2)

            if confidence > 90 and user==1:
                cv2.putText(image, "Hey User1", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )
                whatsapp(input("Enter the Text you want to send: "), input("Enter the phone number to whom you want to send: "))
                mail(input("Enter the text you want to send: "), input("Enter the email to whom you want to send: "), input("Enter your email id from which you want to send: "))
                break

            if confidence > 80 and user==2:
                cv2.putText(image, "Hey User1", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
                cv2.imshow('Face Recognition', image )
                aws_func()
                break

            else:

                cv2.putText(image, "I dont know you", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
                cv2.imshow('Face Recognition', image )

        except:
            cv2.putText(image, "No Face Found", (220, 120) , cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.putText(image, "looking for face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Face Recognition', image )
            pass

        if cv2.waitKey(1) == 13: #13 is the Enter Key
            break

    cap.release()
    cv2.destroyAllWindows()     
    
    
#Main part of program

user1_model=training_model()

# 5 seconds pause between 2 successive face registrations
time.sleep(5)
user2_model=training_model()

process_1=threading.Thread(target=face_recognition, args=(user1_model, 1,))
process_2=threading.Thread(target=face_recognition, args=(user2_model, 2,))

process_1.start()
process_2.start()
