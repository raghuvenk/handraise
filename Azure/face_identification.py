import cognitive_face as CF
import requests
from io import BytesIO
from PIL import Image, ImageDraw

    # status_code: 429
    # code: RateLimitExceeded
    # message: Rate limit is exceeded. Try again later.


KEY = 'fa2df2ad4231452ca04072c796621d10'
BASE_URL = 'https://westcentralus.api.cognitive.microsoft.com/face/v1.0'  # 
PERSON_GROUP_ID = 'b'
CF.BaseUrl.set(BASE_URL)
CF.Key.set(KEY)


CF.person_group.create(PERSON_GROUP_ID, 'hello')

name = "Clemens Siebler"
user_data = 'More information can go here'
response = CF.person.create(PERSON_GROUP_ID, name, user_data)



# Get person_id from response
person_id = response['personId']
CF.person.add_face('https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg', PERSON_GROUP_ID, person_id)

print (CF.person.lists(PERSON_GROUP_ID))

CF.person_group.train(PERSON_GROUP_ID)
response = CF.person_group.get_status(PERSON_GROUP_ID)
status = response['status']




######################################################3
####C SHarp code###############
# faceServiceClient = new FaceServiceClient(KEY)

# # // Create an empty PersonGroup
# string personGroupId = "myfriends"
# await faceServiceClient.CreatePersonGroupAsync(personGroupId, "My Friends")

# # // Define Anna
# CreatePersonResult friend1 = await faceServiceClient.CreatePersonAsync(
#     # // Id of the PersonGroup that the person belonged to
#     personGroupId,    
#     # // Name of the person
#     "Anna"            
# )

# # // Define Bill and Clare in the same way

# # // Directory contains image files of Anna
# string friend1ImageDir = "D:\Pictures\MyFriends\Anna"

# foreach (string imagePath in Directory.GetFiles(friend1ImageDir, "*.jpg"))
#     using (Stream s = File.OpenRead(imagePath))
#     {
#         # // Detect faces in the image and add to Anna
#         await faceServiceClient.AddPersonFaceAsync(
#             personGroupId, friend1.PersonId, s)
#     }
# }
# # // Do the same for Bill and Clare


#######################################################################
#######################################################################
#######################################################################
#######################################################################
#######################################################################


## Training the data structure

CF.person_group.train(PERSON_GROUP_ID)

response = CF.person_group.get_status(PERSON_GROUP_ID)
status = response['status']
# prints as 
# {




##############################################
######## get faces from test file#############
##############################################


testImage = CF.face.detect('https://raw.githubusercontent.com/Microsoft/Cognitive-Face-Windows/master/Data/detection1.jpg')
testFaceIDs = [d['faceId'] for d in testImage]
print (testFaceIDs)



################################################################
################## Verification ################################
################################################################

identified_faces = CF.face.identify(testFaceIDs, PERSON_GROUP_ID)
print (identified_faces)
result_faces = []
i = 0
for face in identified_faces:
    if len(face['candidates']) == 0:
        print("Face Not identified!")
    else: 
        result_faces.append((testImage[i], face['candidates'][0]))
        i = i + 1

allHandraisers = CF.person.lists(PERSON_GROUP_ID)

for i in range(len(allHandraisers)):
    (tempImg, tempID) = result_faces[i]
    for person in allHandraisers:
        if(person['personId'] == tempID):
            allHandraisers[i] = (tempImg, person['name'])
print(allHandraisers)

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


# ## identification of test file
# string testImageFile = ""

# using (Stream s = File.OpenRead(testImageFile))
# {
#     var faces = await faceServiceClient.DetectAsync(s)
#     var faceIds = faces.Select(face => face.FaceId).ToArray()

#     var results = await faceServiceClient.IdentifyAsync(personGroupId, faceIds)
#     for var identifyResult in results:
#         Console.WriteLine("Result of face: {0}", identifyResult.FaceId)
#         if (identifyResult.Candidates.Length == 0):
#             Console.WriteLine("No one identified")
#         else:
#             # // Get top 1 among all candidates returned
#             var candidateId = identifyResult.Candidates[0].PersonId
#             var person = await faceServiceClient.GetPersonAsync(personGroupId, candidateId)
#             Console.WriteLine("Identified as {0}", person.Name)
# }
