#ifndef _PLUGIN_FACTORY_DETECTOR_H_
#define _PLUGIN_FACTORY_DETECTOR_H_

#include <map>
#include <cstring>
#include <cassert>
#include <vector>

// TensorRT
#include <NvCaffeParser.h>
#include <NvInferPlugin.h>

// Custom Layer
#include "interp_layer.h"
#include<iostream>
using namespace nvinfer1;
using namespace nvcaffeparser1;
using namespace plugin;
namespace bdavs {
    class PluginFactoryDetector : public nvinfer1::IPluginFactory, public nvcaffeparser1::IPluginFactory {
    public:
        // caffe parser plugin implementation
        bool isPlugin(const char *name) override {
            return (!strcmp(name, "layer1-upsample")
                    || !strcmp(name, "layer2-upsample")
//                || !strcmp(name, "layer1-route")
//
//                || !strcmp(name, "conf1_perm")
//                || !strcmp(name, "conf2_perm")
//                || !strcmp(name, "conf3_perm")
//                || !strcmp(name, "conf4_perm")
//                || !strcmp(name, "conf5_perm")
//                || !strcmp(name, "conf6_perm")
//
//                || !strcmp(name, "loc1_perm")
//                || !strcmp(name, "loc2_perm")
//                || !strcmp(name, "loc3_perm")
//                || !strcmp(name, "loc4_perm")
//                || !strcmp(name, "loc5_perm")
//                || !strcmp(name, "loc6_perm")
//
//                || !strcmp(name, "conf1_flat")
//                || !strcmp(name, "conf2_flat")
//                || !strcmp(name, "conf3_flat")
//                || !strcmp(name, "conf4_flat")
//                || !strcmp(name, "conf5_flat")
//                || !strcmp(name, "conf6_flat")
//
//                || !strcmp(name, "loc1_flat")
//                || !strcmp(name, "loc2_flat")
//                || !strcmp(name, "loc3_flat")
//                || !strcmp(name, "loc4_flat")
//                || !strcmp(name, "loc5_flat")
//                || !strcmp(name, "loc6_flat")
//
//                || !strcmp(name, "mbox1_priorbox")
//                || !strcmp(name, "mbox2_priorbox")
//                || !strcmp(name, "mbox3_priorbox")
//                || !strcmp(name, "mbox4_priorbox")
//                || !strcmp(name, "mbox5_priorbox")
//                || !strcmp(name, "mbox6_priorbox")
//
//                || !strcmp(name, "mbox_loc_concat")
//                || !strcmp(name, "mbox_conf_concat")
//                || !strcmp(name, "mbox_priorbox_concat")
//                || !strcmp(name, "mbox_conf_reshape")
////                || !strcmp(name, "mbox_conf_softmax")
//                || !strcmp(name, "mbox_conf_flat")
//
//                || !strcmp(name, "detection_out")
//                || !strcmp(name, "detection_out2")
            );
        }

        // serialization plugin implementation
        virtual IPlugin *createPlugin(const char *layerName, const Weights *weights, int nbWeights) override {
            // there's no way to pass parameters through from the model definition,
            // so we have to define it here explicitly
            if (!strcmp(layerName, "layer1-upsample")
                || !strcmp(layerName, "layer2-upsample")) {
                _nvPlugins[layerName] = (plugin::INvPlugin * )(new InterpLayer<38,38>());
                return _nvPlugins.at(layerName);
            } else {
                assert(0);
                return nullptr;
            }
        }

        // deserialization plugin implementation
        IPlugin *createPlugin(const char *layerName, const void *serialData, size_t serialLength) override {
            std::cout<<"layerName:"<<layerName<<std::endl;
            if (!strcmp(layerName, "layer1-upsample")
                || !strcmp(layerName, "layer2-upsample")) {
                _nvPlugins[layerName] = (plugin::INvPlugin * )(new InterpLayer<38,38>(serialData, serialLength));
                return _nvPlugins.at(layerName);
            } else {
                assert(0);
                return nullptr;
            }
        }

        void destroyPlugin() {
            for (auto it = _nvPlugins.begin(); it != _nvPlugins.end(); ++it) {
                it->second->destroy();
                _nvPlugins.erase(it);
            }
        }

    private:
        std::map<std::string, INvPlugin *> _nvPlugins;
    };
}

#endif //_PLUGIN_FACTORY_DETECTOR_H_
