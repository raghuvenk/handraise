import cognitive_face as CF
import requests
from io import BytesIO
from PIL import Image, ImageDraw

KEY = 'fa2df2ad4231452ca04072c796621d10'

faceServiceClient = new FaceServiceClient(KEY)

# // Create an empty PersonGroup
string personGroupId = "myfriends"
await faceServiceClient.CreatePersonGroupAsync(personGroupId, "My Friends")

# // Define Anna
CreatePersonResult friend1 = await faceServiceClient.CreatePersonAsync(
    # // Id of the PersonGroup that the person belonged to
    personGroupId,    
    # // Name of the person
    "Anna"            
)

# // Define Bill and Clare in the same way




# // Directory contains image files of Anna
string friend1ImageDir = "D:\Pictures\MyFriends\Anna"

foreach (string imagePath in Directory.GetFiles(friend1ImageDir, "*.jpg"))
    using (Stream s = File.OpenRead(imagePath))
    {
        # // Detect faces in the image and add to Anna
        await faceServiceClient.AddPersonFaceAsync(
            personGroupId, friend1.PersonId, s)
    }
}
# // Do the same for Bill and Clare







## Training the data structure
await faceServiceClient.TrainPersonGroupAsync(personGroupId)


TrainingStatus trainingStatus = null
while(true):
    trainingStatus = await faceServiceClient.GetPersonGroupTrainingStatusAsync(personGroupId)

    if (trainingStatus.Status != Status.Running):
        break
    await Task.Delay(1000)



## identification of test file
string testImageFile = ""

using (Stream s = File.OpenRead(testImageFile))
{
    var faces = await faceServiceClient.DetectAsync(s)
    var faceIds = faces.Select(face => face.FaceId).ToArray()

    var results = await faceServiceClient.IdentifyAsync(personGroupId, faceIds)
    for var identifyResult in results:
        Console.WriteLine("Result of face: {0}", identifyResult.FaceId)
        if (identifyResult.Candidates.Length == 0):
            Console.WriteLine("No one identified")
        else:
            # // Get top 1 among all candidates returned
            var candidateId = identifyResult.Candidates[0].PersonId
            var person = await faceServiceClient.GetPersonAsync(personGroupId, candidateId)
            Console.WriteLine("Identified as {0}", person.Name)
}