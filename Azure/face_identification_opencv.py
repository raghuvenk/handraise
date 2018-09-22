import os
import cognitive_face as CF
import requests
from io import BytesIO
from PIL import Image, ImageDraw
import numpy as np
import cv2
from random import randint



directory = "C:/Users/alexk/OneDrive/Desktop/cmu_18_fall/hackathon/database/teamprofile/"


#########################################################
################  Azure training ########################
#########################################################

KEY = 'fa2df2ad4231452ca04072c796621d10'
BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0'  # 
PERSON_GROUP_ID = 'handraisecrewfinal5'
CF.BaseUrl.set(BASE_URL)
CF.Key.set(KEY)
############################################################################
# This should be run only once!!!!!!!!!!!!!!!!!!!!!!!!!!!
# CF.person_group.create(PERSON_GROUP_ID, 'Team_HandRaise')
# Names = ["Alex", "Jimin", "kalvin", "raghu"]
# responseList = []
# user_data = "";

# for name in Names:
#     responseList.append(CF.person.create(PERSON_GROUP_ID, name, user_data))
# #############################################################################

# print(CF.person.lists(PERSON_GROUP_ID))
# # This should be run only once!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #This should happen only once at the start of the class.


# #####################################################
# ########## add images for train to group id #########
# #####################################################


# for filename in os.listdir(directory):
#     for i in (CF.person.lists(PERSON_GROUP_ID)):
#     	if(i['name'] in filename):
# 	        person_id = i['personId']
# 	        imageFile = directory + filename
# 	        CF.person.add_face(imageFile, PERSON_GROUP_ID, person_id)
        
#         # 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'

# print (CF.person.lists(PERSON_GROUP_ID))



# #####################################################
# ############ Training the data structure ############
# #####################################################

CF.person_group.train(PERSON_GROUP_ID)

response = CF.person_group.get_status(PERSON_GROUP_ID)
status = response['status']



# ##############################################
# ######## get faces from test file#############
# ##############################################


cap = cv2.VideoCapture(0)

success,image = cap.read()
count = 0


while(success):
    # if pressed q break
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Capture frame-by-frame
    succes, image = cap.read()

    img1 = directory + "frame" + str(count) + ".jpg"
    #save fram as jpeg file
    cv2.imwrite(img1, image)

    # Display the resulting frame
    cv2.imshow('frame', image)
    
    # # Capture image from frame
    # img1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # print(img1)


    testImage = CF.face.detect(img1)

    # 'https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg'
    testFaceIDs = [d['faceId'] for d in testImage]
    # print (testFaceIDs)



    ################################################################
    ################## Verification ################################
    ################################################################

    identified_faces = CF.face.identify(testFaceIDs, PERSON_GROUP_ID)
    # print (identified_faces)
    result_faces = []
    i = 0
    for face in identified_faces:
        if len(face['candidates']) == 0:
            print("Face Not identified!")
        else: 
            result_faces.append((testImage[i], face['candidates'][0]))
            i = i + 1

    classmate = CF.person.lists(PERSON_GROUP_ID)
    questionStudent = []
    # print(classmate)
    # print(result_faces)
    for i in range(len(result_faces)):
        (tempImg, personalInfo) = result_faces[i]
        for person in classmate:
            if(person['personId'] == personalInfo['personId']):
                 questionStudent.append((personalInfo['confidence'], person['name']))
    print(questionStudent)
    count += 1
    ###########
    # RESULTS #
    # [  
    #   {  
    #     u'faceId':u'4fe8f504-8f1d-4e40-8b8a-afe2455b60d6',
    #     u'candidates':[]
    #   },
    #   {  
    #     u'faceId':u'3f5a33d4-f3bc-41f3-8632-1615413e2475',
    #     u'candidates':[  
    #       {  
    #         u'personId':u'15a4e1bf-9dd7-4443-b7eb-3c672e2aa138',
    #         u'confidence':0.73545
    #       }
    #     ]
    #   }
    # ]



#Close video file or capturing device
cap.release()
cv2.destroyAllWindows()

