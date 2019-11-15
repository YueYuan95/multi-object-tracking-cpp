#ifndef _PLUGIN_FACTORY_H_
#define _PLUGIN_FACTORY_H_

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
#include "PReluLayer.h"
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
namespace bdavs {
class PluginFactoryOther : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory{
public:
    // caffe parser plugin implementation
    bool isPlugin(const char* name) override {
        return (!strcmp(name, "layer1-upsample")||
                !strcmp(name, "layer2-upsample")||
                !strcmp(name, "layer3-upsample")||
                !strcmp(name, "layer4-upsample")||
                !strcmp(name, "layer5-upsample")||
                !strcmp(name, "layer6-upsample")||
                !strcmp(name, "layer7-upsample")||
                !strcmp(name, "layer8-upsample")
                );
    }

    // serialization plugin implementation
    virtual IPlugin* createPlugin(const char* layerName, const Weights* weights, int nbWeights) override {
        // there's no way to pass parameters through from the model definition,
        // so we have to define it here explicitly
        if (!strcmp(layerName,"layer1-upsample")||!strcmp(layerName,"layer2-upsample")||
        !strcmp(layerName,"layer3-upsample")||!strcmp(layerName,"layer4-upsample")||
        !strcmp(layerName,"layer5-upsample")||!strcmp(layerName,"layer6-upsample"))
        {
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new InterpLayer<40,24>());
            return _nvPlugins.at(layerName);
        }
        if (!strcmp(layerName,"layer7-upsample"))
        {
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new InterpLayer<80,48>());
            return _nvPlugins.at(layerName);
        }
        if (!strcmp(layerName,"layer8-upsample"))
        {
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new InterpLayer<160,96>());
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
        if (!strcmp(layerName,"layer1-upsample")||!strcmp(layerName,"layer2-upsample")||
        !strcmp(layerName,"layer3-upsample")||!strcmp(layerName,"layer4-upsample")||
        !strcmp(layerName,"layer5-upsample")||!strcmp(layerName,"layer6-upsample"))
        {
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new InterpLayer<40,24>(serialData, serialLength));
            return _nvPlugins.at(layerName);
        }
        if (!strcmp(layerName,"layer7-upsample"))
        {
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new InterpLayer<80,48>(serialData, serialLength));
            return _nvPlugins.at(layerName);
        }
        if (!strcmp(layerName,"layer8-upsample"))
        {
            _nvPlugins[layerName] = (plugin::INvPlugin*)(new InterpLayer<160,96>(serialData, serialLength));
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
#endif //_PLUGIN_FACTORY_HPP_
