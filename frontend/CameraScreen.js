// frontend/CameraScreen.js
import React from 'react';
import { View, StyleSheet } from 'react-native';
import CameraView from './components/CameraView';

export default function CameraScreen() {
  return (
    <View style={styles.container}>
      <CameraView />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000',
  },
});
