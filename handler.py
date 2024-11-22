from PIL import Image
import torch
import boto3
import os
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1

os.environ["TORCH_HOME"] = "/tmp/torch"
mtcnn = MTCNN(image_size=240, margin=0, min_face_size=20) # initializing mtcnn for face detection
resnet = InceptionResnetV1(pretrained='vggface2').eval() # initializing resnet for face img to embeding conversion

def face_recognition_function(key_path):
    # Face extraction
    img = cv2.imread(key_path, cv2.IMREAD_COLOR)
    print(cv2.IMREAD_COLOR)
    print("img",img)
    boxes, _ = mtcnn.detect(img)

    # Face recognition
    key = os.path.splitext(os.path.basename(key_path))[0].split(".")[0]
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    face, prob = mtcnn(img, return_prob=True, save_path=None)
    saved_data = torch.load('/tmp/data.pt')  # loading data.pt file
    if face != None:
        emb = resnet(face.unsqueeze(0)).detach()  # detech is to make required gradient false
        embedding_list = saved_data[0]  # getting embedding data
        name_list = saved_data[1]  # getting list of names
        dist_list = []  # list of matched distances, minimum distance is used to identify the person
        for idx, emb_db in enumerate(embedding_list):
            dist = torch.dist(emb, emb_db).item()
            dist_list.append(dist)
        idx_min = dist_list.index(min(dist_list))

        # Save the result name in a file
        with open("/tmp/" + key + ".txt", 'w+') as f:
            f.write(name_list[idx_min])
        return name_list[idx_min]
    else:
        print(f"No face is detected")
    return
s3 = boto3.client('s3')
def handler(event, context):	
    bucket = event['bucket_name']
    
    key = event['file_name']
   
    output_bucket = "1229609006-output"
    key1 = 'data.pt' 
    o_path = os.path.join('/tmp',key)
   
    o_path2 = os.path.join('/tmp',key1)
    bucket_name = '1229609006-databucketforface'
    s3.download_file(bucket,key,o_path)
    try:
        s3.download_file(bucket_name, key1,o_path2)
        print("downloaded data.pt file")
    except Exception as e:
        print("unable to download data.pt")
    print(os.listdir('/tmp'))
    output =  face_recognition_function(o_path)
    print("output:",output)
    file_name_without_extension = key.split('.')[0]
    text_file_name = file_name_without_extension + ".txt"
    file_path = text_file_name
    with open("/tmp/" + file_path, "w") as file:
        file.write(output)
    try:
        s3.upload_file("/tmp/" + file_path,output_bucket,file_path)
        print("Successfully uploaded to s3 Bucket")
    except Exception as e:
        print(f"Failed to upload {output} to S3 bucket {output_bucket}: {str(e)}")