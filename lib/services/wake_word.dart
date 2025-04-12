import 'dart:async';
import 'package:flutter/material.dart';
import 'package:porcupine_flutter/porcupine_manager.dart';
import 'package:porcupine_flutter/porcupine_error.dart';
import 'package:permission_handler/permission_handler.dart';

class WakeWordService {
  static PorcupineManager? _porcupineManager;
  static bool _isListening = false;
  
  // Callback to be executed when wake word is detected
  static VoidCallback? onWakeWordDetected;
  
  // Picovoice API key
  static const String _apiKey = "LTXoX0LP4WKj7eZNJFN2OKH95LcbvL3bMu2IV59f2Tc8atCY54cavQ==";
  
  // Path to your custom keyword file (stored in assets)
  static const String _customKeywordPath = "assets/hey_grab.ppn";
static void _logDebug(String message) {
  debugPrint('🎙️ Wake Word Service: $message');
}

  static Future<bool> initialize() async {
  try {
    // Request microphone permission
    _logDebug('Initializing...');
    if (!await _requestPermissions()) {
      debugPrint('Wake Word: Microphone permission denied');
      return false;
    }
    _logDebug('Microphone permission granted');
    _logDebug('Wake Word: Attempting to initialize with key: ${_apiKey.substring(0, 5)}...');
    _logDebug('Wake Word: Looking for model at: $_customKeywordPath');
    
    // Create Porcupine Manager with custom keyword file
_porcupineManager = await PorcupineManager.fromKeywordPaths(
      _apiKey,
      [_customKeywordPath],
      (keywordIndex) {
        _logDebug('✅ WAKE WORD DETECTED! Index: $keywordIndex');
        if (onWakeWordDetected != null) {
          _logDebug('Executing callback');
          onWakeWordDetected!();
        }
      },
      errorCallback: (error) {
        _logDebug('❌ ERROR: ${error.message}');
      },
    );
    
    debugPrint('✅ Wake word service initialized successfully!');
    return true;
  } catch (err) {
    debugPrint('❌ Failed to initialize Porcupine: $err');
    return false;
  }
}
  
  static Future<bool> _requestPermissions() async {
    final status = await Permission.microphone.request();
    return status.isGranted;
  }
  
  static void _wakeWordCallback(int keywordIndex) {
    debugPrint('Wake word "Hey, Grab" detected!');
    if (onWakeWordDetected != null) {
      onWakeWordDetected!();
    }
  }
  
  static void _errorCallback(PorcupineException error) {
    debugPrint('Porcupine error: ${error.message}');
  }
  
  static Future<bool> startListening() async {
    if (_porcupineManager == null) {
      if (!await initialize()) {
        return false;
      }
    }
    
    try {
      if (!_isListening) {
        await _porcupineManager?.start();
        _isListening = true;
        debugPrint('Wake word detection started');
      }
      return true;
    } catch (e) {
      debugPrint('Failed to start wake word detection: $e');
      return false;
    }
  }
  
  static Future<void> stopListening() async {
    try {
      if (_isListening && _porcupineManager != null) {
        await _porcupineManager?.stop();
        _isListening = false;
        debugPrint('Wake word detection stopped');
      }
    } catch (e) {
      debugPrint('Failed to stop wake word detection: $e');
    }
  }
  
  static Future<void> dispose() async {
    try {
      await _porcupineManager?.delete();
      _porcupineManager = null;
      _isListening = false;
    } catch (e) {
      debugPrint('Error disposing wake word service: $e');
    }
  }
  
  static bool isListening() {
    return _isListening;
  }
}