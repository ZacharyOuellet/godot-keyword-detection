#include "register_types.h"

#include "mfcc_processor.h"
#include "pncc_processor.h"
#include "dtw_matcher.h"

#include <gdextension_interface.h>
#include <godot_cpp/core/defs.hpp>
#include <godot_cpp/godot.hpp>

using namespace godot;

void initialize_godot_keyword_detection_types(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
    ClassDB::register_class<MFCCProcessor>();
    ClassDB::register_class<PNCCProcessor>();
    ClassDB::register_class<DTWMatcher>();
}

void uninitialize_godot_keyword_detection_types(ModuleInitializationLevel p_level) {
    if (p_level != MODULE_INITIALIZATION_LEVEL_SCENE) {
        return;
    }
}

extern "C" {

    GDExtensionBool GDE_EXPORT godot_keyword_detection_library_init(
        GDExtensionInterfaceGetProcAddress p_get_proc_address,
        const GDExtensionClassLibraryPtr p_library,
        GDExtensionInitialization* r_initialization) {

        godot::GDExtensionBinding::InitObject init_obj(p_get_proc_address, p_library, r_initialization);
        init_obj.register_initializer(initialize_godot_keyword_detection_types);
        init_obj.register_terminator(uninitialize_godot_keyword_detection_types);
        init_obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);

        return init_obj.init();
    }

} // extern "C"
