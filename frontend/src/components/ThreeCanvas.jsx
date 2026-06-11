import React, { Suspense } from 'react';
import { Canvas } from '@react-three/fiber';

export const ThreeCanvas = ({ children, className = '', cameraPosition = [0, 0, 7] }) => {
  // If custom z-index is provided in className, skip the default -z-10 background z-index
  const hasZIndex = /z-\d+/.test(className) || className.includes('-z-') || className.includes('z-auto');
  const zClass = hasZIndex ? '' : '-z-10';

  return (
    <div className={`w-full h-full absolute inset-0 overflow-hidden ${zClass} ${className}`}>
      <Canvas
        camera={{ position: cameraPosition, fov: 70 }}
        gl={{ antialias: true, alpha: true, powerPreference: 'high-performance' }}
        dpr={[1, 1.5]}
      >
        <fog attach="fog" args={['#030014', 10, 30]} />
        <Suspense fallback={null}>
          {children}
        </Suspense>
      </Canvas>
    </div>
  );
};

export default ThreeCanvas;
