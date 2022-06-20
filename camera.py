import tensorflow as tf
import sys
import os
import cv2



def main():
    key=0
    cam = cv2.VideoCapture(0)
    label2="Press S to capture"
    res=""

    
    while True:
        ret, img = cam.read()
        cv2.imwrite("img.jpg",img)
        if key==1:
            cnt=0
            # Disable tensorflow compilation warnings
            os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
            import tensorflow as tf
            
            
            
            image_path = "input.jpg"
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
                    print(human_string)
                    score = predictions[0][node_id]
                    if cnt==0:
                        res=label_lines[node_id]
                        res2=score*100
                        res2=str(res2)
                        res2=res2[0:4]
                        cnt=cnt+1
                        print ("Last result was ",res,res2)
                    print('%s (score = %.5f)' % (human_string, score))
                    label2="Press S to capture"
                    key=0
        font = cv2.FONT_HERSHEY_SIMPLEX
          
        # org
        org = (0, 20)
        org2 = (0, 50)  
        # fontScale
        fontScale = 1
           
        # Blue color in BGR
        color = (255, 0, 0)
          
        # Line thickness of 2 px
        thickness = 1            
        cv2.putText(img, label2, org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)            
        cv2.putText(img, "result is "+res, org2, font, 
                   fontScale, color, thickness, cv2.LINE_AA)        
        cv2.imshow('leaf scan', img)
        k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
        if k == 27:
            break
        if k == ord('s'):
            key=1
        #return res
    print("\n [INFO] Exiting Program and cleanup stuff")
    cam.release()
    cv2.destroyAllWindows()
    return res,res2
#main()
