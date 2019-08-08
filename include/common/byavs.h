/*
 * byavs.h
 *
 *  Created on: 2019年4月30日
 *      Author: Dell
 */

#ifndef TOOLS_SO_INCLUDE_BYAVS_API_H_
#define TOOLS_SO_INCLUDE_BYAVS_API_H_
#include <opencv2/core/core.hpp>
#include <vector>
#include <stdlib.h>
#include <memory>
namespace byavs{
#define DEEP_FEATURE_FACE_LEN   (128)
#define DEEP_FEATURE_LEN        (2048)
#define PLATE_LENGTH            (10)
//#include <NetOperator.h>


typedef struct
{
  unsigned char* data;
  int height;
  int width;
  int channels;
} GpuMat;

typedef struct BboxInfo
{
  int topLeftX;
  int topLeftY;
  int width;
  int height;
}BboxInfo;

typedef struct {
  float confidence;
  int minWidth;
  int minHeight;
  int maxWidth;
  int maxHeight;
} BasicDetectParas;
typedef std::vector<GpuMat> GpuImgBGRArray;
typedef std::vector<cv::Mat> CpuImgBGRArray;

/**
 * @brief The Detector class
 * Input:GpuImgBGRArray
 * Output:objArray
 * Batch can be adjusted freely
 */
typedef struct {
  BasicDetectParas detectPara;
  //Other required parameters can be added.
} DetectParas;

typedef struct
{
  int label;
  float score;
  BboxInfo box;
} DetectObject;

typedef struct{
  int label;
  float score;
  cv::Rect_<float> bbox;
} DetectRectObject;

typedef std::vector<DetectObject> DetectObjects;
typedef std::vector<DetectObjects> ObjArray;
class Detector
{
public:
  // init detector
  bool init(const std::string& model_dir, const DetectParas& pas, const int gpu_id);

  //img inference
  bool inference(const GpuImgBGRArray& imgBGRs, ObjArray& objects);
  bool inference(const CpuImgBGRArray& imgBGRs, ObjArray& objects);

  //detector release
  void release();

private:
    void * detector;
};

/**
 * @brief The MultiTracker class
 * Input1:cv::Mat
 * Input2:detectObjects
 * Output:std::vector<TrackeKeyObject>
 */
typedef struct {
    //The parameters of MultiTracker.

} TrackeParas;

/*
typedef struct {
    int label;
    BboxInfo box;  // key box
    cv::Mat img; // full image corresponding to the optimal frame
} TrackeKeyObject;
*/
typedef struct {
  int camID;
  int channelID;
  GpuMat gpuImg;
  DetectObjects objs;
} TrackeInputGPU;

typedef struct {
  int camID;
  int channelID;
  cv::Mat cpuImg;
  DetectObjects objs;
} TrackeInputCPU;

typedef struct {                                                                                                                                                                
  int id;                                                                                                                                                                     
  int label;                                                                                                                                                                  
  BboxInfo box;                                                                                                                                                               
  GpuMat gpuImg;                                                                                                                                                                
  bool return_state;                                                                                                                                                          
} TrackeObjectGPU;

typedef struct {
  int id;                                                                                                                                                                     
  int label;                                                                                                                                                                  
  BboxInfo box;                                                                                                                                                               
  cv::Mat cpuImg;                                                                                                                                                                
  bool return_state;
} TrackeObjectCPU;

/*save for iou_tracker, strongly recommonad do NOT use this*/
typedef struct {
  int camID;
  int channelID;
  GpuMat gpuImg;
  DetectObjects objs;
} TrackeInput;

typedef struct
{
  int label;
  BboxInfo box;  // key box
  GpuMat img; // full image corresponding to the optimal frame
} TrackeKeyObject;

typedef std::vector<TrackeKeyObject> TrackeObjects;
typedef std::vector<TrackeObjects> TrackeResultArray;
typedef std::vector<TrackeInput> TrackeInputArray;

/*Recommand Use followed api*/
typedef std::vector<TrackeInputGPU> TrackeInputGPUArray;
typedef std::vector<TrackeInputCPU> TrackeInputCPUArray;

typedef std::vector<TrackeObjectGPU> TrackeObjectGPUs;
typedef std::vector<TrackeObjectCPU> TrackeObjectCPUs;

typedef std::vector<TrackeObjectGPUs> TrackeResultGPUArray;
typedef std::vector<TrackeObjectCPUs> TrackeResultCPUArray;

class Tracking{

public:
   bool init(const std::string& model_dir, const TrackeParas& pas, const int gpu_id);
   //bool inference(const cv::Mat& imgBGR, const DetectObjects& detectResults, std::vector<TrackeKeyObject>& keyObjects);
   
   //Old inter
   bool inference(const TrackeInputArray& inputs, TrackeResultArray& resultArray);

   bool inference(const TrackeInputGPUArray& inputs, TrackeResultGPUArray& resultArray);
   bool inference(const TrackeInputCPUArray& inputs, TrackeResultCPUArray& resultArray);

   void release();

private:
   void*  trackingPtr;
};


/**
 * @brief The FrameSelector class
 */

typedef struct {
    //The parameters of VehicleRecognizer.
} SelectorParas;

class FrameSelector{

    public:

        bool init(const std::string& model_dir,const SelectorParas& pas,const int gpu_id);

        //PedestrianFeature inference
        bool inference(const CpuImgBGRArray& imgs, std::vector<int>& quilty);
        bool inference(const GpuImgBGRArray& imgs, std::vector<int>& quilty);
        //PedestrianFeature release
        void release();
        
    private:
        void * extractor_model;
};



/**
 * @brief The PersonRecognizer class
 */

typedef struct {
    //The parameters of VehicleRecognizer.
} PedParas;

typedef struct PedestrainAttr
{ 
  char hairstyle;//0-未知、1-短发、2-马尾、3-盘发、4-头部被遮挡、5-长发、6-光头
  char ageGroup;//0-未知、1-幼儿、2-少年、3-青年、4-中年、5-老年
  char upperCategory;//0-马甲吊带背心、1-衬衫、2-西服、3-毛衣、4-皮衣夹克、5-羽绒服、6-大衣风衣、7-连衣裙、8-T恤、9-无上衣、10-其它
  char upperTexture;//0-纯色、1-碎花、2-条纹、3-格子、4-文字、5-其他
  char upperColor;//0-黑、1-白、2-灰、3-绿、4-深灰、5-红、6-黄、7-蓝、8-紫、9-棕、10-混色、11-其他
  char lowerCategory;//0-长裤、1-七分裤、2-长裙、3-短裙、4-短裤、5-连衣裙
  char lowerColor;//0-黑、1-白、2-灰、3-绿、4-深灰、5-红、6-黄、7-蓝、8-紫、9-棕、10-混色、11-其他
  char shoesCategory;//0-未知、1-光脚,2-皮鞋、3-运动鞋、4-靴子、5-凉鞋、6-休闲鞋、7-其他
  char shoesColor;//0-黑、1-白、2-灰、3-绿、4-深灰、5-红、6-黄、7-蓝、8-紫、9-棕、10-混色、11-其他
  char bagCategory;//0-单肩包、1-双肩包、2-挎包、3-其他
  char holdBaby;//0-抱小孩、1-背小孩、3-其他
  char hasHandItems;//0-无手持物、1-有单个手持物 、 2-有多个手持物
  char handItems;//0-手机、1-手拎包、2-拉杆箱、3-水杯、4-婴儿车、5-购物袋、6-其他
  char hatType;//0-帽子、1-头盔、2-未戴帽子
  char hatColor;//0-黑、1-白、2-灰、3-绿、4-深灰、5-红、6-黄、7-蓝、8-紫、9-棕、10-混色、11-其他
  char orientation;//0-正向、1-侧身、2-背部
  char posture;//0-胖、1-瘦、2-中
  char racial;//0-汉族、1-维族、2-黑人、3-白人
  char pedHeight;//0-高、1-中、2-低
  bool hasUmbrella;//有打伞、无打伞
  bool holdPhone;//手持接打电话、未接打电话
  bool hasScarf;//有围巾、无围巾
  bool gender;//ture-男、false-女
  bool hasGlasses;//带眼镜  不带眼镜
  bool hasMask;//戴口罩  不带口罩
  bool hasBag;//有背包、无背包
  bool hasBaby;//有小孩、无小孩
} PedestrainAttr;
typedef std::vector<PedestrainAttr> PedAttrArray;
class PersonStructured {
public:
    // init
    bool init(const std::string& model_dir, const PedParas& pas, const int gpu_id);
    // inference
    bool inference(const CpuImgBGRArray& imgs, PedAttrArray& pedOut);
    bool inference(const GpuImgBGRArray& imgs,PedAttrArray& pedOut);
    //
    void release();
private:
  void* model;

};

/**
 * @brief The pedestrian feature class
 * Input:GpuImgBGRArray
 * Output:float**
 */
typedef struct {
  int FEATURE_LEN;
    //The parameters of pedestrian feature.
} PedFeatureParas;

//float pedFeature[BANTCH_SIZE][DEEP_FEATURE_LEN];//Specific dimensions need to be given.

class PedestrianFeature {
public:
    // init PedestrianFeature
    bool init(const std::string& model_dir,const PedFeatureParas& pas,const int gpu_id);

    //PedestrianFeature inference
    bool inference(const CpuImgBGRArray& imgs, float** pedFeatures);
    bool inference(const GpuImgBGRArray &imgs, float ** pedFeatures);
    //PedestrianFeature release
    void release();
private:
  void * extractor_model;
};



/**
 * @brief The face Detector class
 * Input:GpuImgBGRArray
 * Output:FaceObjArray
 */
typedef struct {
  BasicDetectParas detectPara;
  //Other required parameters can be added.
} FaceDetectParas;

typedef struct
{
  float score;
  BboxInfo box;
} FaceDetectObject;

typedef std::vector<FaceDetectObject> FaceDetectObjects;
typedef std::vector<FaceDetectObjects> FaceObjArray;

class FaceDetector
{
public:
  // init detector
  bool init(const std::string& model_dir, const FaceDetectParas& pas, const int gpu_id);

  //img inference
  bool inference(const CpuImgBGRArray& imgBGRs, FaceObjArray& objects);
    bool inference(const GpuImgBGRArray& imgBGRs, FaceObjArray& objects);
  //detector release
  void release();
private:
  void * detector;
};

/**
 * @brief The FaceRecognizer class
 * Input:ImgBGRArray
 * Output:faceAttrArray
 */
typedef struct {
    //The parameters of FaceRecognizer.
} FaceParas;

typedef struct FaceAttr
{
  bool gender;
  char hairstyle;
  char ageGroup;
  char facialExpression;
  bool mask;
  bool glasses;
  char hat;
  char hatColor;
  bool beard;
  char direction;
} FaceAttr;
typedef std::vector<FaceAttr> FaceAttrArray;
class FaceStructured {
public:
    // init face Recognizer
    bool init(const std::string& model_dir, const FaceParas& pas, const int gpu_id);

  //face Recognizer inference
    bool inference(const GpuImgBGRArray& imgs, FaceAttrArray& faceOut);

    //face Recognizer release
    void release();
};

/**
 * @brief The face feature class
 * Input:ImgBGRArray
 * Output:float**
 */
typedef struct {
  int FEATURE_LEN;
    //The parameters of face feature.
} FaceFeatureParas;

//float faceFeature[BANTCH_SIZE][DEEP_FEATURE_LEN];//Specific dimensions need to be given.

class FaceFeature {
public:
    // init FaceFeature
    bool init(const std::string& model_dir, const FaceFeatureParas& pas,const int gpu_id);

    //FaceFeature inference
    bool inference(const CpuImgBGRArray& imgs, float** faceFeatures);
    bool inference(const GpuImgBGRArray &imgs, float ** faceFeatures);
    //FaceFeature release
    void release();
private:
    void * model;
};



/**
 * @brief The VehicleRecognizer class
 * Input:ImgBGRArray
 * Output:vehAttrArray
 */
typedef struct {
    //The parameters of VehicleRecognizer.
} VehParas;

typedef struct VehicleAttr
{
  char vehicleSide;
  char vehicleCategory;
  char vehicleColor;
  bool annualInspecStandard;
  bool pendant;
  bool decoration;
  bool driverSafetyBelt;
  bool copilotSafetyBelt;
  bool tissueBox;
  bool driverPhone;
  bool sunShield;
  bool skylightPeople;
  unsigned int  vehicleBrand;
} VehAttr;
typedef std::vector<VehAttr> VehAttrArray;
class VehicleStructured {
public:
    // init Vehicle Recognizer
    bool init(const std::string& model_dir, const VehParas& pas, const int gpu_id);

    //Vehicle Recognizer inference 因为需要使用标志物检测进行再次分析，所以输入加入标志物检测结果
    bool inference(const CpuImgBGRArray& imgs,  VehAttrArray& vehOut);
    bool inference(const GpuImgBGRArray& imgs,  VehAttrArray& vehOut);
    //Vehicle Recognizer release
    void release();
private:
  void *model;
};


/**
 * @brief The vehicle feature class
 * Input:ImgBGRArray
 * Output:float**
 */
typedef struct {
  int FEATURE_LEN;
    //The parameters of vehicle feature.
} VehFeatureParas;

//float vehFeature[BANTCH_SIZE][DEEP_FEATURE_LEN];//Specific dimensions need to be given.

class VehicleFeature {
public:
    // init VehicleFeature
    bool init(const std::string& model_dir,const VehFeatureParas& pas,const int gpu_id);

    //VehicleFeature inference
    bool inference(const CpuImgBGRArray& imgs, float** vehFeatures);
    bool inference(const GpuImgBGRArray& imgs, float** vehFeatures);
    //VehicleFeature release
    void release();
private:
  void * extractor_model;
};


/**
 * @brief The plate Detector class
 * Input:ImgBGRArray
 * Output:MarkObjArray
 */
typedef struct {
  BasicDetectParas detectPara;
  //Other required parameters can be added.
} MarkDetectParas;

typedef struct
{
  int label;//0-车牌 、1-人(驾驶员和副驾驶)、2-年检标、3-后视镜、4-车大灯、5-中控摆件、6-遮阳板、7-挂饰、8-车窗
  float score;
  BboxInfo box;
} MarkDetectObject;
typedef std::vector<MarkDetectObject> MarkDetectObjects;
typedef std::vector<MarkDetectObjects> MarkObjArray;
class MarkDetector //标志物检测器(车牌、挂件、摆件等9中检测目标)
{
public:
  // init markDetector
  bool init(const std::string& model_dir, const MarkDetectParas& pas, const int gpu_id);

  //markDetector inference
  bool inference(const CpuImgBGRArray& imgBGRs, MarkObjArray& objects);
  bool inference(const GpuImgBGRArray& imgBGRs, MarkObjArray& objects);
  //markDetector release
  void release();
private:
    void * detector;
};




/**
 * @brief The plateRecognizer class
 * Input:GpuImgBGRArray
 * Output:plateAttrArray
 */
typedef struct {
    //The parameters of FaceRecognizer.
} PlateParas;

typedef struct PlateAttr
{
  char plateCategory;
  char plateColor;
  char hasPlate;
  char plateNumber[PLATE_LENGTH];//Specific dimensions need to be given.
}PlateAttr;
typedef std::vector<PlateAttr> PlateAttrArray;
class PlateStructured {
public:
    // init plate Recognizer
    bool init(const std::string& model_dir, const PlateParas& pas, const int gpu_id);

    //plate Recognizer inference
    bool inference(const CpuImgBGRArray& imgs, PlateAttrArray& plateOut);
    bool inference(const GpuImgBGRArray& imgs, PlateAttrArray& plateOut);
    //plate Recognizer release
    void release();
public:
    void *model;
};






/**
 * @brief The nonVehicleRecognizer class
 * Input:ImgBGRArray
 * Output:nonVehAttrArray
 */
typedef struct {
    //The parameters of nonVehicleRecognizer.
} NonVehParas;

typedef struct NonVehicle
{
  char hairstyle;
  char ageGroup;
  char upperCategory;
  char upperTexture;
  char upperColor;
  char hatType;
  char hatColor;
  char nonMotorType;
  char nonMotorColor;
  char nonMotorPassenger;
  char nonMotorFrontDirection;
  bool gender;
  bool hasMask;
  bool hasGlasses;
  bool sunShade;
  bool nonMotorBoot;
}NonVehAttr;
typedef std::vector<NonVehAttr> NonVehAttrArray;
class NonVehicleStructured {
public:
    // init nonVehicle Recognizer
    bool init(const std::string& model_dir,const NonVehParas& pas, const int gpu_id);

    //nonVehicle Recognizer inference
    bool inference(const GpuImgBGRArray& imgs, NonVehAttrArray& nonVehOut);
    bool inference(const CpuImgBGRArray& imgs, NonVehAttrArray& nonVehOut);

    //nonVehicle Recognizer release
    void release();
};

/**
 * @brief The nonvehicle feature class
 * Input:GpuImgBGRArray
 * Output:float**
 */
typedef struct {
  int FEATURE_LEN;
    //The parameters of nonvehicle feature.
} NonVehFeatureParas;

//float nonVehFeature[BANTCH_SIZE][DEEP_FEATURE_LEN];//Specific dimensions need to be given.

class NonVehFeature {
public:
    // init NonVehFeature
    bool init(const std::string& model_dir,const NonVehFeatureParas& pas,const int gpu_id);

    //NonVehFeature inference
    bool inference(const GpuImgBGRArray& img, float** nonVehFeatures);
    bool inference(const CpuImgBGRArray& img, float** nonVehFeatures);
    //NonVehFeature release
    void release();
};
}
#endif /* TOOLS_SO_INCLUDE_BYAVS_API_H_ */
