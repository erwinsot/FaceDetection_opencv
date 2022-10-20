#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <time.h> 
#include <pthread.h>
#include <omp.h>
#define THREADS 1
#define PAD 8

using namespace cv;
using namespace std;
void DifFace();
string cascadeName, nestedCascadeName;
double scale=1;
CascadeClassifier cascade, nestedCascade;
Mat imgResl[301];
VideoWriter writer, wrSalida;
int ITERATIONS;
VideoCapture cap;
int  i;
stringstream file,file2,entradaReselsed;
int main(int argc, char** argv)
{
    clock_t t;
    t=clock();
	struct Video *vid;
	Mat frame, image;
    ofstream ofs("Data.txt");	
	nestedCascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye_tree_eyeglasses.xml");
    cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml");	
	cap.open(argv[1]);
    int frame_width = cap.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = cap.get(CAP_PROP_FRAME_HEIGHT); 
    Size frame_size(frame_width, frame_height);
    int fcc=cv::VideoWriter::fourcc('X','V','I','D');
    double fps = cap.get(CAP_PROP_FPS);    
    writer = VideoWriter(argv[2] + std::string(".avi"),fcc,fps,frame_size,true);
    wrSalida= VideoWriter(argv[2] + std::string(".avi"),fcc,fps,frame_size,true);         
    ITERATIONS=cap.get(CAP_PROP_FRAME_COUNT);
    int total_frames = cap.get(CAP_PROP_FRAME_COUNT);       
	  if (cap.isOpened())
            {
                        // Capturando cada frame del video
                        cout << "Face Detection is started...." << endl;
                        while (1)
                        {
                                    cap >> frame;
                                    double num_frame=cap.get(CAP_PROP_POS_FRAMES);                                   
                                    if (frame.empty()){
                                        if (!ofs.bad())
                                            {
                                                t = clock()-t;
                                                double time = (double(t)/CLOCKS_PER_SEC);
                                                ofs << "El tiempo de ejcucion es" << endl;
                                                ofs << time << endl;
                                                ofs.close();
                                            }

                                                break;
                                    }
                                    //Almacenando cada fram en carpeta
                                    stringstream file;
                                    file<<"/home/erwin/openmp/frames/frame"<<num_frame<<".jpg";
                                    cv::imwrite(file.str(),frame);
                                    Mat frame1 = frame.clone();
                                    cv::resize(frame1, frame1, Size(1000, 700), 0, 0, INTER_CUBIC);                                    
                                    //
                                    writer.write(frame);                                     
                                    
                                    // Para mostar video 

                                    imshow("Face Detection", frame1);

                                    
                                    
                                    char c = (char)waitKey(10);
                                    // Press q to exit from the window
                                    if (c == 27 || c == 'q' || c == 'Q')
                                                break;
                        }
            }
	else
    cout << "Could not Open Video/Camera! ";


    

    #pragma omp parallel num_threads(THREADS)
        {
            
            DifFace();
        }  

    
    
    for(i=1;i<total_frames;i++)
    {

        entradaReselsed<<"/home/erwin/openmp/framesDif/frame"<<i<<".jpg";
        Mat im = imread(entradaReselsed.str(), IMREAD_COLOR);

        if(im.empty())
        {
            std::cout << "Could not read the image fimal: "<< std::endl;
            entradaReselsed.str("");
            
            continue;
            
        }
        imgResl[i]=im;
        
        entradaReselsed.str("");
    }

    cout<<"zona no paralela"<<endl;
    for(i=1;i<sizeof imgResl;i++){
        if(imgResl[i].empty())
        {
            continue;
        }
        wrSalida.write(imgResl[i]);
    }                                        
                                    
   
    return 0;


}
//funcion a paralelizar 

void DifFace()
{
    int tid =omp_get_thread_num();    
    int initIteration, endIteration, threadId = tid;
    initIteration = (ITERATIONS/THREADS) * threadId;
    cout <<"desde el hilo "<< tid<< "el ini es "<< initIteration<< endl;
    if(initIteration==0)
        initIteration++;
    endIteration = initIteration + ((ITERATIONS/THREADS) );
    cout <<"desde el hilo "<< tid<< "el end es "<< endIteration<< endl;    
    vector<Rect> faces; 
    Mat gray, smallImg; 
    double fx = 1 / scale;
    stringstream entrada, salida;
    //tomamos cada frame para difuminar la parte del rostro
    while (initIteration<endIteration)
    {
        entrada<<"/home/erwin/openmp/frames/frame"<<initIteration<<".jpg";        
        Mat img = imread(entrada.str(), IMREAD_COLOR);
        if(img.empty())
        {
            std::cout << "Could not read the image: "<< std::endl;
          entrada.str("");
            initIteration++;
            continue;
            
        }
        cvtColor(img, gray, COLOR_BGR2GRAY);
        resize(gray, smallImg, Size(),fx, fx, INTER_LINEAR);
        equalizeHist(smallImg, smallImg);
        cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        for (size_t i = 0; i < faces.size(); i++)
        {
            Rect r = faces[i];
            Mat smallImgROI;
            vector<Rect> nestedObjects;                                                
            Scalar color = Scalar(255, 0, 0); 
                                                           
                                                                    
            // dibujar cuadrado de cara
            rectangle(img, Point(cvRound(r.x * scale), cvRound(r.y * scale)),Point(cvRound((r.x + r.width - 1) * scale),cvRound((r.y + r.height - 1) * scale)), color, 3, 8, 0);
             
             //algoritmo para difuminar el rostro                                                            
             cv::Point topLeft = cv::Point(cvRound(r.x * scale), cvRound(r.y * scale));
             cv::Point bottomRight = cv::Point(cvRound((r.x + r.width - 1) * scale),cvRound((r.y + r.height - 1) * scale));
             cv::Rect roi = cv::Rect(topLeft, bottomRight);                               

             cv::GaussianBlur(img(roi), img(roi), cv::Size(91, 91), 0);                                                       
             if (nestedCascade.empty())
                continue;
             smallImgROI = smallImg(r);
        }
        salida<<"/home/erwin/openmp/framesDif/frame"<<initIteration<<".jpg";
        cv::imwrite(salida.str(),img);        
        salida.str("");
        entrada.str("");
        initIteration++;

    }
}

