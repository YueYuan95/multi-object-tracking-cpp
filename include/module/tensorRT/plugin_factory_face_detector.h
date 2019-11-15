#ifndef _PLUGIN_FACTORY_FACE_DETECTOR_H_
#define _PLUGIN_FACTORY_FACE_DETECTOR_H_

#include <map>
#include <cstring>
#include <cassert>
#include <vector>

// TensorRT
#include <NvCaffeParser.h>
#include <NvInferPlugin.h>

// Custom Layer
#include "interp_layer.h"
#include "reshape_layer.h"
#include "softmax_layer.h"
#include "flatten_layer.h"

using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;

namespace bdavs {
class PluginFactoryFaceDetector : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory{
public:
    // caffe parser plugin implementation
    bool isPlugin(const char* name) override {
        return ( !strcmp(name, "layer1-upsample")
                || !strcmp(name, "layer2-upsample")
            );
    }

    // serialization plugin implementation
    virtual IPlugin* createPlugin(const char* layerName, const Weights* weights, int nbWeights) override {
        // there's no way to pass parameters through from the model definition,
        // so we have to define it here explicitly
        if (!strcmp(layerName, "layer1-upsample")
                            || !strcmp(layerName, "layer2-upsample"))
        {
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new InterpLayer<38,38>());
            return _nvPlugins.at(layerName);
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    // deserialization plugin implementation
    IPlugin* createPlugin(const char* layerName, const void* serialData, size_t serialLength) override {
        if (!strcmp(layerName, "layer1-upsample")
                            || !strcmp(layerName, "layer2-upsample"))
        {
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new InterpLayer<38,38>(serialData, serialLength));
            return _nvPlugins.at(layerName);
        }
        else
        {
            assert(0);
            return nullptr;
        }
    }

    void destroyPlugin() {
        for (auto it = _nvPlugins.begin(); it != _nvPlugins.end(); ++it)
        {
            it->second->destroy();
            _nvPlugins.erase(it);
        }
    }

private:
    std::map<std::string, INvPlugin*> _nvPlugins;
};

}
#endif //_PLUGIN_FACTORY_FACE_DETECTOR_H_
