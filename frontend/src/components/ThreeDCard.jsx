import React, { useRef } from 'react';
import { motion, useMotionValue, useSpring, useTransform } from 'framer-motion';

export const ThreeDCard = ({ children, className = '' }) => {
  const cardRef = useRef(null);

  // Set up motion values for x/y mouse coordinates relative to card center
  const x = useMotionValue(0);
  const y = useMotionValue(0);

  // Dampen the movements using springs for smoother interactions
  const springConfig = { damping: 25, stiffness: 120, mass: 0.8 };
  const rotateXSpring = useSpring(y, springConfig);
  const rotateYSpring = useSpring(x, springConfig);

  // Transform coordinates to degrees of rotation (max 15 degrees)
  const rotateX = useTransform(rotateXSpring, [-0.5, 0.5], [15, -15]);
  const rotateY = useTransform(rotateYSpring, [-0.5, 0.5], [-15, 15]);

  const handleMouseMove = (event) => {
    if (!cardRef.current) return;
    const rect = cardRef.current.getBoundingClientRect();
    
    // Get mouse position relative to card boundaries (value between 0 and 1)
    const relativeX = (event.clientX - rect.left) / rect.width;
    const relativeY = (event.clientY - rect.top) / rect.height;

    // Shift origin to center (value between -0.5 and 0.5)
    x.set(relativeX - 0.5);
    y.set(relativeY - 0.5);
  };

  const handleMouseLeave = () => {
    // Reset rotations when mouse exits
    x.set(0);
    y.set(0);
  };

  return (
    <div 
      className="perspective-1000 flex items-center justify-center"
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
    >
      <motion.div
        ref={cardRef}
        style={{
          rotateX,
          rotateY,
          transformStyle: 'preserve-3d',
        }}
        className={`glass-panel rounded-2xl p-6 relative w-full h-full border border-white/10 shadow-2xl transition-shadow duration-300 hover:shadow-purple-500/10 ${className}`}
      >
        {/* Glow Overlay */}
        <div 
          className="absolute inset-0 rounded-2xl opacity-0 hover:opacity-10 pointer-events-none transition-opacity duration-500"
          style={{
            background: 'radial-gradient(circle at center, rgba(124,58,237,0.4) 0%, transparent 70%)',
            transform: 'translateZ(1px)',
          }}
        />
        {/* Content */}
        <div style={{ transform: 'translateZ(30px)' }}>
          {children}
        </div>
      </motion.div>
    </div>
  );
};
export default ThreeDCard;
