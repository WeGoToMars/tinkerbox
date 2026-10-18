// APFS clone staging for Core AI's external weight constants.
#import <Foundation/Foundation.h>
#import <objc/message.h>
#import <objc/runtime.h>
#include <copyfile.h>
#include <dlfcn.h>
#include <stdbool.h>
#include <unistd.h>

typedef BOOL (*CompileFunction)(id, SEL, id, id, unsigned, void *);
static CompileFunction originalCompile;

static NSDictionary *constants(id model) {
    return ((id (*)(id, SEL))objc_msgSend)(model, NSSelectorFromString(@"externConstants"));
}

static void fail(const char *message) {
    fprintf(stderr, "%s\n", message);
    _exit(2);
}

static BOOL compileWithClones(id self, SEL command, id model, id options, unsigned qos, void *error) {
    NSDictionary *external = constants(model);
    NSDictionary *files = external[@"FileConstants"];
    NSDictionary *environment = NSProcessInfo.processInfo.environment;
    NSString *destination = environment[@"COW_CLIENT_DEST"];
    NSString *prefix = [environment[@"COW_ASSET_PREFIX"] stringByAppendingString:@"/"];
    if (!destination || !prefix) fail("missing clone staging paths");
    if (!files.count) return originalCompile(self, command, model, options, qos, error);

    for (id key in files) {
        if (![files[key][@"Path"] hasPrefix:prefix]) {
            // Framework auxiliary graphs may have small resources outside the bundle.
            for (id candidate in files) {
                NSDictionary *attributes = [NSFileManager.defaultManager
                    attributesOfItemAtPath:files[candidate][@"Path"] error:nil];
                if (!attributes || attributes.fileSize > 64 * 1024 * 1024)
                    fail("unknown large external resource");
            }
            return originalCompile(self, command, model, options, qos, error);
        }
    }

    NSMutableDictionary *rewritten = [NSMutableDictionary dictionary];
    NSMutableDictionary *paths = [NSMutableDictionary dictionary];
    for (id key in files) {
        NSDictionary *entry = files[key];
        NSString *source = entry[@"Path"];
        NSString *target = paths[source];
        if (!target) {
            target = [destination stringByAppendingPathComponent:
                [NSString stringWithFormat:@"resources-%@.bin", NSUUID.UUID.UUIDString]];
            copyfile_state_t state = copyfile_state_alloc();
            if (!state) fail("cannot allocate clone state");
            if (copyfile(source.fileSystemRepresentation, target.fileSystemRepresentation,
                         state, COPYFILE_DATA | COPYFILE_NOFOLLOW | COPYFILE_CLONE_FORCE)) {
                perror("clone");
                _exit(2);
            }
            bool cloned = false;
            copyfile_state_get(state, COPYFILE_STATE_WAS_CLONED, &cloned);
            copyfile_state_free(state);
            if (!cloned) fail("weight resource was not cloned");
            paths[source] = target;
        }
        NSMutableDictionary *replacement = [entry mutableCopy];
        replacement[@"Path"] = target;
        [replacement removeObjectForKey:@"_SBExtension"];
        rewritten[key] = replacement;
    }
    NSMutableDictionary *replacement = [external mutableCopy];
    replacement[@"FileConstants"] = rewritten;
    ((void (*)(id, SEL, id))objc_msgSend)(model, NSSelectorFromString(@"setExternConstants:"), replacement);
    id result = ((id (*)(id, SEL, id, BOOL, id))objc_msgSend)(NSClassFromString(@"_ANECloneHelper"),
        NSSelectorFromString(@"cloneIfWritable:isEncryptedModel:cloneDirectory:"), model, NO,
        [destination stringByAppendingPathComponent:@"preflight"]);
    if (!result || ![constants(model)[@"FileConstants"] isEqual:rewritten])
        fail("ANE clone helper changed external paths");
    BOOL ok = originalCompile(self, command, model, options, qos, error);
    if (!ok) {
        id failure = error ? (__bridge id)*(void **)error : nil;
        fprintf(stderr, "ANE compilation failed: %s\n", failure ? [failure description].UTF8String : "unknown");
        _exit(4);
    }
    return ok;
}

void installStaging(void) {
    dlopen("/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine", RTLD_NOW);
    Method method = class_getInstanceMethod(NSClassFromString(@"_ANEClient"),
        NSSelectorFromString(@"compileModel:options:qos:error:"));
    if (!method) fail("ANE compile method unavailable");
    originalCompile = (CompileFunction)method_setImplementation(method, (IMP)compileWithClones);
}
