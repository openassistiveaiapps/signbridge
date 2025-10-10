// frontend/components/CameraView.js
import React, { useState, useEffect, useRef } from 'react';
import { View, Text, StyleSheet, Button } from 'react-native';
import { Camera } from 'expo-camera';

export default function CameraView() {
  const [hasPermission, setHasPermission] = useState(null);
  const cameraRef = useRef(null);

  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);

  if (hasPermission === null) {
    return <Text style={styles.info}>Requesting camera permission...</Text>;
  }
  if (hasPermission === false) {
    return <Text style={styles.info}>No access to camera</Text>;
  }

  return (
    <Camera
      style={styles.camera}
      type={Camera.Constants.Type.front}
      ref={cameraRef}
    >
      <View style={styles.overlay}>
        <Text style={styles.overlayText}>Camera Ready 👌</Text>
      </View>
    </Camera>
  );
}

const styles = StyleSheet.create({
  camera: {
    flex: 1,
  },
  overlay: {
    position: 'absolute',
    bottom: 50,
    alignSelf: 'center',
  },
  overlayText: {
    color: '#fff',
    fontSize: 18,
  },
  info: {
    flex: 1,
    color: '#fff',
    textAlign: 'center',
    textAlignVertical: 'center',
    backgroundColor: '#000',
  },
});
