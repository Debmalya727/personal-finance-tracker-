import React, { useRef, useState } from 'react';
import { useFrame } from '@react-three/fiber';
import { Text } from '@react-three/drei';
import * as THREE from 'three';

const AssetBar = ({ position, height, color, name, value }) => {
  const meshRef = useRef();
  const matRef = useRef();
  const [hovered, setHovered] = useState(false);

  useFrame(() => {
    if (!meshRef.current || !matRef.current) return;
    const targetScaleY = hovered ? 1.15 : 1.0;
    const targetScaleXZ = hovered ? 1.1 : 1.0;
    meshRef.current.scale.y = THREE.MathUtils.lerp(meshRef.current.scale.y, targetScaleY, 0.1);
    meshRef.current.scale.x = THREE.MathUtils.lerp(meshRef.current.scale.x, targetScaleXZ, 0.1);
    meshRef.current.scale.z = THREE.MathUtils.lerp(meshRef.current.scale.z, targetScaleXZ, 0.1);
    matRef.current.emissiveIntensity = THREE.MathUtils.lerp(
      matRef.current.emissiveIntensity,
      hovered ? 0.7 : 0.15,
      0.1
    );
  });

  return (
    <group position={position}>
      <mesh
        ref={meshRef}
        onPointerOver={(e) => { e.stopPropagation(); setHovered(true); }}
        onPointerOut={() => setHovered(false)}
        position={[0, height / 2, 0]}
      >
        <boxGeometry args={[0.8, height, 0.8]} />
        <meshStandardMaterial
          ref={matRef}
          color={color}
          roughness={0.15}
          metalness={0.85}
          emissive={color}
          emissiveIntensity={0.15}
        />
      </mesh>

      {/* Floating Text Label */}
      <Text
        position={[0, height + 0.6, 0]}
        fontSize={0.22}
        color="#ffffff"
        anchorX="center"
        anchorY="middle"
        outlineWidth={0.02}
        outlineColor="#000000"
      >
        {`${name}\n₹${value}`}
      </Text>
    </group>
  );
};

export const ThreeDChart = ({ data = [] }) => {
  const groupRef = useRef();

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.getElapsedTime() * 0.12) * 0.4;
    }
  });

  if (!data || data.length === 0) return null;

  const maxVal = Math.max(...data.map((d) => d.value), 1);
  const chartData = data.map((d, index) => {
    const height = (d.value / maxVal) * 3.0 + 0.5;
    const xPos = (index - (data.length - 1) / 2) * 1.6;
    return { ...d, height, position: [xPos, -1.5, 0] };
  });

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[0, 5, 5]} intensity={2} color="#7c3aed" />
      <pointLight position={[-5, 2, 0]} intensity={1} color="#06b6d4" />
      <group ref={groupRef}>
        {chartData.map((bar) => (
          <AssetBar
            key={bar.name}
            position={bar.position}
            height={bar.height}
            color={bar.color}
            name={bar.name}
            value={bar.value.toLocaleString()}
          />
        ))}
        <gridHelper args={[10, 10, '#302660', '#181236']} position={[0, -1.5, 0]} />
      </group>
    </>
  );
};

export default ThreeDChart;
