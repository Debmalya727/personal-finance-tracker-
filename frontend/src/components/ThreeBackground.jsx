import React, { useRef, useMemo } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Animated floating torus knots
const FloatingShape = ({ position, color, scale, speed }) => {
  const meshRef = useRef();
  useFrame((state) => {
    if (!meshRef.current) return;
    const t = state.clock.getElapsedTime();
    meshRef.current.rotation.x = t * speed * 0.3;
    meshRef.current.rotation.y = t * speed * 0.5;
    meshRef.current.position.y = position[1] + Math.sin(t * speed + position[0]) * 0.3;
  });
  return (
    <mesh ref={meshRef} position={position} scale={scale}>
      <torusKnotGeometry args={[1, 0.3, 64, 8]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={0.4}
        roughness={0.1}
        metalness={0.8}
        transparent
        opacity={0.6}
        wireframe={false}
      />
    </mesh>
  );
};

// Animated particle field with thousands of stars
const ParticleField = ({ count = 2000 }) => {
  const points = useRef();

  const [positions, colors] = useMemo(() => {
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);
    const colorPalette = [
      new THREE.Color('#7c3aed'),
      new THREE.Color('#06b6d4'),
      new THREE.Color('#db2777'),
      new THREE.Color('#a78bfa'),
      new THREE.Color('#38bdf8'),
    ];

    for (let i = 0; i < count; i++) {
      pos[i * 3] = (Math.random() - 0.5) * 30;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 30;
      pos[i * 3 + 2] = (Math.random() - 0.5) * 20;

      const c = colorPalette[Math.floor(Math.random() * colorPalette.length)];
      col[i * 3] = c.r;
      col[i * 3 + 1] = c.g;
      col[i * 3 + 2] = c.b;
    }
    return [pos, col];
  }, [count]);

  useFrame((state) => {
    if (!points.current) return;
    points.current.rotation.y = state.clock.getElapsedTime() * 0.03;
    points.current.rotation.x = state.clock.getElapsedTime() * 0.01;
  });

  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[positions, 3]} />
        <bufferAttribute attach="attributes-color" args={[colors, 3]} />
      </bufferGeometry>
      <pointsMaterial
        size={0.06}
        vertexColors
        transparent
        opacity={0.8}
        sizeAttenuation
      />
    </points>
  );
};

// Glowing energy ring
const EnergyRing = ({ position, color, rotationSpeed }) => {
  const ring = useRef();
  useFrame((state) => {
    if (!ring.current) return;
    const t = state.clock.getElapsedTime();
    ring.current.rotation.x = t * rotationSpeed;
    ring.current.rotation.z = t * rotationSpeed * 0.7;
  });

  return (
    <mesh ref={ring} position={position}>
      <torusGeometry args={[2, 0.02, 16, 100]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={2}
        transparent
        opacity={0.5}
      />
    </mesh>
  );
};

export const ThreeBackground = () => {
  return (
    <>
      <ambientLight intensity={0.2} />
      <pointLight position={[0, 0, 5]} intensity={2} color="#7c3aed" />
      <pointLight position={[-5, 5, 0]} intensity={1.5} color="#06b6d4" />
      <pointLight position={[5, -5, 0]} intensity={1.5} color="#db2777" />

      <ParticleField count={1500} />

      <FloatingShape position={[-6, 2, -3]} color="#7c3aed" scale={0.5} speed={0.3} />
      <FloatingShape position={[6, -2, -2]} color="#06b6d4" scale={0.4} speed={0.4} />
      <FloatingShape position={[0, -4, -5]} color="#db2777" scale={0.35} speed={0.2} />

      <EnergyRing position={[0, 0, -3]} color="#7c3aed" rotationSpeed={0.1} />
      <EnergyRing position={[3, 2, -4]} color="#06b6d4" rotationSpeed={0.15} />
      <EnergyRing position={[-3, -2, -4]} color="#db2777" rotationSpeed={0.12} />
    </>
  );
};

export default ThreeBackground;
