import tensorflow as tf
import sys
import os
import cv2
totalcnt1=0
def main(img,plastic):
    global totalcnt1
    cnt=0
    # Disable tensorflow compilation warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    import tensorflow as tf

    image_path = img#"cardboard1.jpg"
    #image_path=file

    # Read the image_data
    image_data = tf.io.gfile.GFile(image_path, 'rb').read()


    # Loads label file, strips off carriage return
    label_lines = [line.rstrip() for line
                       in tf.io.gfile.GFile("logs/output_labels.txt")]

    # Unpersists graph from file
    with tf.io.gfile.GFile("logs/output_graph.pb", 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.compat.v1.Session() as sess:
        # Feed the image_data as input to the graph and get first prediction
        softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

        predictions = sess.run(softmax_tensor, \
                 {'DecodeJpeg/contents:0': image_data})

        # Sort to show labels of first prediction in order of confidence
        top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]

        for node_id in top_k:
            human_string = label_lines[node_id]
            #print(human_string)
            score = predictions[0][node_id]
            if cnt==0:
                res=label_lines[node_id]
                res2=score*100
                res2=str(res2)
                res2=res2[0:4]
                cnt=cnt+1
                print ("prediction is ",res," with score ",res2)
                if res in plastic:
                    print(str(img)[:-4]+" is Plastic")
                    totalcnt1=totalcnt1+1
                    res="plastic"
                else:
                    print(str(img)[:-4]+" is Non Plastic")
                    res="Non Plastic"
            #print('%s (score = %.5f)' % (human_string, score))
        return res,res2
plastic=["plastic","plastic covers","plastic tubs"]
j=0
tot=0
for item in ["metal71.jpg","input.jpg"]:
    print("")
    tot=tot+1
    print("testing ",item)
    res,res2=main(item,plastic)
    frame=cv2.imread(item)
    font = cv2.FONT_HERSHEY_SIMPLEX
  
    # org
    org = (5,20)
      
    # fontScale
    fontScale = 1
       
    # Blue color in BGR
    color = (255, 0, 0)
      
    # Line thickness of 2 px
    thickness = 1
       
    # Using cv2.putText() method
    frame = cv2.putText(frame, 'Category: '+res, org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    color = (0, 255, 0)
    org = (5, 50)
    frame = cv2.putText(frame, 'Plastic Count : '+str(totalcnt1), org, font, 
                       fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow(str(item)[:-4],frame)
    cv2.waitKey(0)
print("total count of plastic is:",totalcnt1)
print("total number of images tested is:",tot)  
print("total plastic count percentage is ",totalcnt1,"/",tot,"=",(totalcnt1/tot)*100,"%")
result="total plastic count percentage is "+str(totalcnt1)+"/"+str(tot)+"="+str((totalcnt1/tot)*100)+" %"
frame=cv2.imread("result.jpg")
font = cv2.FONT_HERSHEY_SIMPLEX
  
# org
org = (5, 30)
  
# fontScale
fontScale = 1
   
# Blue color in BGR
color = (255, 0, 0)
  
# Line thickness of 2 px
thickness = 2
   
# Using cv2.putText() method
frame = cv2.putText(frame, "total plastic cnt: "+str(totalcnt1), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
color = (0, 255, 0)
org = (5, 90)
frame = cv2.putText(frame, "total images tested: "+str(tot), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
color = (0, 0, 255)
org = (5, 150)
frame = cv2.putText(frame, result, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
cv2.imshow("final result",frame)
cv2.waitKey(0)