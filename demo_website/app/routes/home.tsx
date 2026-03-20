import React, { useEffect, useRef, useState } from 'react';
import hands_package from '@mediapipe/hands';
import * as cam from '@mediapipe/camera_utils';
import draw_package from '@mediapipe/drawing_utils';

const { Hands, HAND_CONNECTIONS } = hands_package;
const { drawConnectors, drawLandmarks } = draw_package;

const ProjectPage: React.FC = () => {
  return (
    <div style={pageStyle}>
      {/* --- HERO SECTION (The 2% Summary) --- */}
      <header style={heroStyle}>
        <h1 style={titleStyle}>Intuitive Robot Teleoperation</h1>
        <p style={subtitleStyle}>
          Bridging the gap between natural human motion and robotic precision 
          through Vision-Based Mapping and Shared Autonomy.
        </p>
        <div style={badgeContainer}>
          <span style={badgeStyle}>MediaPipe</span>
          <span style={badgeStyle}>Robosuite</span>
          <span style={badgeStyle}>Shared Autonomy</span>
        </div>
      </header>

      <main style={mainContentStyle}>
        
        {/* --- LIVE DEMO SECTION --- */}
        <section style={sectionStyle}>
          <div style={cardStyle}>
            <h3 style={sectionTitleStyle}>Interactive Hand-Tracking Module</h3>
            <p style={textStyle}>
              This module demonstrates the <strong>Given</strong> input: a real-time monocular 
              webcam feed. We estimate depth by measuring the Euclidean distance between 
              the index and pinky joints, providing a robust Z-axis coordinate without specialized depth sensors.
            </p>
            <HandTracker />
          </div>
        </section>

        {/* --- VIDEO DEMO SECTION (The 3% Component) --- */}
        <section style={sectionStyle}>
        <h3 style={{ textAlign: "center", fontSize: "1.8rem", fontWeight:"700" }}>Keyboard Teleop vs. Manual Teleop  </h3>
              <video
                controls
                style={{ width: '100%', borderRadius: '8px' }}
              >
                <source src="/cs188final/timeComparison.mp4" type="video/mp4" />
                Your browser does not support the video tag.
              </video>
        </section>

        {/* --- METHODOLOGY SECTION --- */}
        <section style={gridSectionStyle}>
          <div style={infoCardStyle}>
            <h4>Vision-Based Mapping</h4>
            <p>Translating 2D camera coordinates to 3D robot workspace using dynamic scaling and joint-based depth estimation.</p>
          </div>
          <div style={infoCardStyle}>
            <h4>Shared Autonomy</h4>
            <p>A biasing protocol that activates within a 0.15m radius of target objects, allowing for fine-tuned precision during the "grab" phase.</p>
          </div>
        </section>

      </main>

      <footer style={footerStyle}>
        University Robotics Project — 2026
      </footer>
    </div>
  );
};

/* --- YOUR ORIGINAL LOGIC (Integrated) --- */
const HandTracker: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isLoaded, setIsLoaded] = useState<boolean>(false);

  useEffect(() => {
    const hands = new Hands({
      locateFile: (file: string) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5,
    });

    hands.onResults((results) => {
      if (!canvasRef.current || !videoRef.current) return;
      setIsLoaded(true);
      const canvasCtx = canvasRef.current.getContext('2d');
      if (!canvasCtx) return;

      canvasCtx.save();
      canvasCtx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
      canvasCtx.drawImage(results.image, 0, 0, canvasRef.current.width, canvasRef.current.height);

      if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
        const landmarks = results.multiHandLandmarks[0];
        drawConnectors(canvasCtx, landmarks, HAND_CONNECTIONS, { color: '#00FF00', lineWidth: 5 });
        drawLandmarks(canvasCtx, landmarks, { color: '#FF0000', lineWidth: 2 });
      }
      canvasCtx.restore();
    });

    if (videoRef.current) {
      const camera = new cam.Camera(videoRef.current, {
        onFrame: async () => { if (videoRef.current) await hands.send({ image: videoRef.current }); },
        width: 640, height: 480,
      });
      camera.start();
    }
    return () => hands.close();
  }, []);

  return (
    <div style={{ position: 'relative', width: '100%', maxWidth: '640px', marginTop: '20px' }}>
      {!isLoaded && <div style={loaderStyle}>Initializing MediaPipe...</div>}
      <video ref={videoRef} style={{ display: 'none' }} playsInline muted />
      <canvas ref={canvasRef} width={640} height={480} style={canvasStyle} />
    </div>
  );
};

/* --- STYLES (Clean & Professional) --- */
const pageStyle: React.CSSProperties = {
  backgroundColor: '#f8fafc',
  minHeight: '100vh',
  color: '#1e293b',
  fontFamily: 'system-ui, -apple-system, sans-serif',
  lineHeight: '1.6'
};

const heroStyle: React.CSSProperties = {
  backgroundColor: '#0f172a',
  color: 'white',
  padding: '80px 20px',
  textAlign: 'center',
};

const titleStyle: React.CSSProperties = { fontSize: '3rem', fontWeight: 800, marginBottom: '16px' };
const subtitleStyle: React.CSSProperties = { fontSize: '1.25rem', maxWidth: '800px', margin: '0 auto', color: '#94a3b8' };

const badgeContainer: React.CSSProperties = { display: 'flex', gap: '10px', justifyContent: 'center', marginTop: '20px' };
const badgeStyle: React.CSSProperties = { backgroundColor: '#334155', padding: '6px 12px', borderRadius: '20px', fontSize: '0.8rem' };

const mainContentStyle: React.CSSProperties = { maxWidth: '1000px', margin: '0 auto', padding: '40px 20px' };
const sectionStyle: React.CSSProperties = { marginBottom: '60px' };
const sectionTitleStyle: React.CSSProperties = { fontSize: '1.8rem', fontWeight: 700, marginBottom: '20px', color: '#0f172a' };

const cardStyle: React.CSSProperties = { 
  backgroundColor: 'white', padding: '30px', borderRadius: '16px', 
  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)', display: 'flex', flexDirection: 'column', alignItems: 'center' 
};

const videoPlaceholderStyle: React.CSSProperties = {
  width: '100%', aspectRatio: '16/9', backgroundColor: '#e2e8f0', 
  borderRadius: '16px', display: 'flex', flexDirection: 'column', 
  justifyContent: 'center', alignItems: 'center', border: '2px dashed #cbd5e1'
};

const playIconStyle: React.CSSProperties = { fontSize: '50px', color: '#64748b', marginBottom: '10px' };

const gridSectionStyle: React.CSSProperties = { display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' };
const infoCardStyle: React.CSSProperties = { backgroundColor: 'white', padding: '20px', borderRadius: '12px', borderTop: '4px solid #3b82f6' };

const canvasStyle: React.CSSProperties = { width: '100%', height: 'auto', borderRadius: '12px', boxShadow: '0 10px 15px -3px rgb(0 0 0 / 0.1)' };
const loaderStyle: React.CSSProperties = { padding: '20px', backgroundColor: '#f1f5f9', borderRadius: '8px', textAlign: 'center' };
const footerStyle: React.CSSProperties = { textAlign: 'center', padding: '40px', color: '#64748b', fontSize: '0.9rem' };

const textStyle: React.CSSProperties = { color: '#475569', textAlign: 'center', maxWidth: '600px' };

export default ProjectPage;