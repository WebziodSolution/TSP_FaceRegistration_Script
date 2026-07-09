import React, { useEffect, useRef, useState } from 'react';
import * as faceapi from 'face-api.js';
import { styled, useTheme } from '@mui/material/styles';
import Components from '../../muiComponents/components';
import Button from '../../common/buttons/button';
import { connect } from 'react-redux';
import { setAlert } from '../../../redux/commonReducers/commonReducers';
import CustomIcons from '../../common/icons/CustomIcons';
import { faceRecognitionAPIBaseURL, faceRecognitionModelURL } from '../../../config/apiConfig/apiConfig';
import { useNavigate } from 'react-router-dom';
import { speakMessage } from '../../../service/common/commonService';

const BootstrapDialog = styled(Components.Dialog)(({ theme }) => ({
    '& .MuiDialogContent-root': {
        padding: theme.spacing(2),
    },
    '& .MuiDialogActions-root': {
        padding: theme.spacing(1),
    },
}));

const API_BASE_URL = faceRecognitionAPIBaseURL;
const modelsPath = faceRecognitionModelURL;

function FaceRegistration({ setAlert, open, handleClose, employeeId, type = null, setLoginInfo }) {
    const theme = useTheme();
    const navigate = useNavigate();

    const onClose = () => {
        handleClose(); // Trigger parent to close modal
        stopWebcam(); // Ensure webcam is stopped
    };

    const faceAlignedRef = useRef(false);
    const countdownIntervalRef = useRef(null);
    const webcamVideoRef = useRef(null);
    const capturedPhotoRef = useRef(null);
    const photoCanvasRef = useRef(null);
    const detectionCanvasRef = useRef(null);
    const faceFrameRef = useRef(null);
    const webcamDisplayRef = useRef(null);
    const capturedDisplayRef = useRef(null);
    const countdownActiveRef = useRef(false);
    const faceDetectionIntervalRef = useRef(null);
    const latestDescriptorRef = useRef(null);
    const livenessVerifiedRef = useRef(false);
    const lastSpokenTimeRef = useRef(0);

    const [currentStream, setCurrentStream] = useState(null);
    const [capturedImageDataURL, setCapturedImageDataURL] = useState(null);
    const [faceDescriptor, setFaceDescriptor] = useState(null);
    const [modelsLoaded, setModelsLoaded] = useState(false);
    const [isPhotoAlreadyCaptured, setIsPhotoAlreadyCaptured] = useState(false); // Manages visibility
    const [isLoading, setIsLoading] = useState(false);
    const [registerMessage, setRegisterMessage] = useState({ text: '', type: '' });

    const dataURLtoBlob = (dataurl) => {
        const arr = dataurl.split(',');
        const mime = arr[0].match(/:(.*?);/)[1];
        const bstr = atob(arr[1]);
        let n = bstr.length;
        const u8arr = new Uint8Array(n); // Corrected typo
        while (n--) {
            u8arr[n] = bstr.charCodeAt(n);
        }
        return new Blob([u8arr], { type: mime });
    };

    const showMessage = (setter, msg, type) => {
        setter({ text: msg, type });
    };

    const clearMessage = (setter) => {
        setter({ text: '', type: '' });
    };

    const loadModels = async () => {
        try {
            await Promise.all([
                faceapi.nets.tinyFaceDetector.load(modelsPath),
                faceapi.nets.faceLandmark68Net.load(modelsPath),
                faceapi.nets.faceRecognitionNet.load(modelsPath)
            ]);
            setModelsLoaded(true);
            console.log('Face-API models loaded successfully.');
            // Initial message after models load, before webcam is fully ready
            showMessage(setRegisterMessage, 'Models loaded. Starting webcam...', 'info');
        } catch (error) {
            console.error('Failed to load face-api.js models:', error);
            showMessage(setRegisterMessage, 'Error loading face detection models. Please refresh.', 'error');
        }
    };

    const handlePlayVideo = async (videoElement) => {
        try {
            await videoElement.play();
        } catch (err) {
            console.log('Video play error:', err);
            videoElement.muted = true;
            try {
                await videoElement.play();
            } catch (err2) {
                console.log('Second video play attempt failed:', err2);
            }
        }
    };

    const detectFaces = async () => {
        if (isPhotoAlreadyCaptured) {
            console.log('detectFaces: Photo already captured, skipping detection cycle.');
            return;
        }

        if (!webcamVideoRef.current || webcamVideoRef.current.paused || webcamVideoRef.current.ended || !modelsLoaded) {
            console.log('detectFaces: Webcam or models not ready.');
            return;
        }

        try {
            const detectionWithDescriptor = await faceapi.detectSingleFace(
                webcamVideoRef.current,
                new faceapi.TinyFaceDetectorOptions()
            ).withFaceLandmarks().withFaceDescriptor();

            if (!detectionWithDescriptor || detectionWithDescriptor.detection.score < 0.6) {
                if (!isPhotoAlreadyCaptured) {
                    showMessage(setRegisterMessage, 'No face detected. Please stand in front of the camera.', 'info');
                }
                faceAlignedRef.current = false;
                if (faceFrameRef.current) {
                    faceFrameRef.current.style.borderColor = "#ef4444"; // Red
                    faceFrameRef.current.querySelectorAll('.corner').forEach(c => c.style.borderColor = "#ef4444");
                }
                if (countdownIntervalRef.current) {
                    clearInterval(countdownIntervalRef.current);
                    countdownIntervalRef.current = null;
                    countdownActiveRef.current = false;
                }
                if (detectionCanvasRef.current) {
                    const ctx = detectionCanvasRef.current.getContext('2d');
                    ctx.clearRect(0, 0, detectionCanvasRef.current.width, detectionCanvasRef.current.height);
                }
                return;
            }

            const displaySize = {
                width: webcamVideoRef.current.offsetWidth,
                height: webcamVideoRef.current.offsetHeight
            };
            const resizedDetection = faceapi.resizeResults(detectionWithDescriptor, displaySize);
            const detection = resizedDetection.detection;
            const landmarks = resizedDetection.landmarks;

            // Draw detection box and landmarks
            if (detectionCanvasRef.current) {
                const ctx = detectionCanvasRef.current.getContext('2d');
                ctx.clearRect(0, 0, detectionCanvasRef.current.width, detectionCanvasRef.current.height);
                // faceapi.draw.drawDetections(detectionCanvasRef.current, detection);
                faceapi.draw.drawFaceLandmarks(detectionCanvasRef.current, landmarks);
            }

            const faceBox = detection.box;
            const frameRect = faceFrameRef.current.getBoundingClientRect();
            const videoRect = webcamVideoRef.current.getBoundingClientRect();

            const frameX = frameRect.left - videoRect.left;
            const frameY = frameRect.top - videoRect.top;
            const frameWidth = frameRect.width;
            const frameHeight = frameRect.height;

            const scaleX = webcamVideoRef.current.videoWidth / webcamVideoRef.current.offsetWidth;
            const scaleY = webcamVideoRef.current.videoHeight / webcamVideoRef.current.offsetHeight;

            const scaledFaceBox = new faceapi.Rect(
                faceBox.x * scaleX,
                faceBox.y * scaleY,
                faceBox.width * scaleX,
                faceBox.height * scaleY
            );

            const scaledFrameX = frameX * scaleX;
            const scaledFrameY = frameY * scaleY;
            const scaledFrameWidth = frameWidth * scaleX;
            const scaledFrameHeight = frameHeight * scaleY;

            // Relaxed margin to allow easier face alignment (especially on phones)
            const relaxedMarginX = scaledFrameWidth * 0.05; // 5%
            const relaxedMarginY = scaledFrameHeight * 0.07; // 7%

            const isWithinFrame =
                scaledFaceBox.x + scaledFaceBox.width > scaledFrameX + relaxedMarginX &&
                scaledFaceBox.y + scaledFaceBox.height > scaledFrameY + relaxedMarginY &&
                scaledFaceBox.x < scaledFrameX + scaledFrameWidth - relaxedMarginX &&
                scaledFaceBox.y < scaledFrameY + scaledFrameHeight - relaxedMarginY;

            if (isWithinFrame) {
                const isLogin = type === "login";
                // Liveness check using head turning (only for login)
                if (isLogin && !livenessVerifiedRef.current) {
                    const noseTip = landmarks.positions[30];
                    const leftEdge = landmarks.positions[0];
                    const rightEdge = landmarks.positions[16];

                    const getEuclideanDistance = (pt1, pt2) => {
                        return Math.sqrt(Math.pow(pt1.x - pt2.x, 2) + Math.pow(pt1.y - pt2.y, 2));
                    };

                    const leftDist = getEuclideanDistance(noseTip, leftEdge);
                    const rightDist = getEuclideanDistance(noseTip, rightEdge);
                    const yawRatio = leftDist / (leftDist + rightDist);

                    if (yawRatio < 0.43 || yawRatio > 0.57) {
                        livenessVerifiedRef.current = true;
                        faceFrameRef.current.style.borderColor = "#22c55e"; // Green
                        faceFrameRef.current.querySelectorAll('.corner').forEach(c => c.style.borderColor = "#22c55e");
                        showMessage(setRegisterMessage, 'Liveness verified! Please look straight.', 'success');
                        speakMessage('Liveness verified! Please look straight.');
                    } else {
                        faceFrameRef.current.style.borderColor = "#eab308"; // Yellow
                        faceFrameRef.current.querySelectorAll('.corner').forEach(c => c.style.borderColor = "#eab308");

                        const statusMsg = "Please TURN HEAD slightly Left or Right";
                        showMessage(setRegisterMessage, statusMsg, 'warning');

                        if (Date.now() - lastSpokenTimeRef.current > 5000) {
                            speakMessage(statusMsg);
                            lastSpokenTimeRef.current = Date.now();
                        }

                        faceAlignedRef.current = false;
                        if (countdownIntervalRef.current) {
                            clearInterval(countdownIntervalRef.current);
                            countdownIntervalRef.current = null;
                            countdownActiveRef.current = false;
                        }
                        return;
                    }
                }

                faceAlignedRef.current = true;
                faceFrameRef.current.style.borderColor = "#22c55e"; // Green
                faceFrameRef.current.querySelectorAll('.corner').forEach(c => c.style.borderColor = "#22c55e");

                // Store the latest face descriptor string
                latestDescriptorRef.current = JSON.stringify(Array.from(resizedDetection.descriptor));

                if (!isPhotoAlreadyCaptured && !countdownActiveRef.current) {
                    countdownActiveRef.current = true;
                    let countdown = 2;
                    countdownIntervalRef.current = setInterval(() => {
                        countdown--;
                        if (countdown > 0) {
                        } else {
                            clearInterval(countdownIntervalRef.current);
                            countdownIntervalRef.current = null;
                            countdownActiveRef.current = false;
                            capturePhoto();
                        }
                    }, 1000);
                }
            } else {
                faceAlignedRef.current = false;
                faceFrameRef.current.style.borderColor = "#ef4444"; // Red
                faceFrameRef.current.querySelectorAll('.corner').forEach(c => c.style.borderColor = "#ef4444");

                if (countdownIntervalRef.current) {
                    clearInterval(countdownIntervalRef.current);
                    countdownIntervalRef.current = null;
                    countdownActiveRef.current = false;
                    showMessage(setRegisterMessage, 'Face moved out of alignment. Please re-center.', 'warning');
                } else if (!isPhotoAlreadyCaptured) {
                    showMessage(setRegisterMessage, 'Please position your face within the frame.', 'info');
                }
            }

        } catch (err) {
            console.error('Face detection error:', err);
        }
    };

    const startWebcam = async () => {
        stopWebcam();
        clearMessage(setRegisterMessage);
        clearMessage(setRegisterMessage);
        setCapturedImageDataURL(null);
        setFaceDescriptor(null);
        latestDescriptorRef.current = null;
        livenessVerifiedRef.current = false;
        lastSpokenTimeRef.current = 0;
        setIsPhotoAlreadyCaptured(false); // Ensure this is false to restart detection
        countdownActiveRef.current = false;

        // --- VISIBILITY CONTROL ---
        if (webcamDisplayRef.current) webcamDisplayRef.current.classList.remove('hidden');
        if (capturedDisplayRef.current) capturedDisplayRef.current.classList.add('hidden');
        // Ensure detection elements are visible when webcam is active
        if (webcamVideoRef.current) webcamVideoRef.current.classList.remove('hidden');
        if (detectionCanvasRef.current) detectionCanvasRef.current.classList.remove('hidden');
        if (faceFrameRef.current) faceFrameRef.current.classList.remove('hidden');


        // Clean up previous stream if exists
        if (currentStream) {
            currentStream.getTracks().forEach(track => {
                track.stop();
            });
            setCurrentStream(null);
        }
        // Also clear any existing face detection interval
        if (faceDetectionIntervalRef.current) {
            clearInterval(faceDetectionIntervalRef.current);
            faceDetectionIntervalRef.current = null;
        }
        if (countdownIntervalRef.current) {
            clearInterval(countdownIntervalRef.current);
            countdownIntervalRef.current = null;
        }


        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 1280 },
                    height: { ideal: 720 },
                    facingMode: 'user'
                }
            });
            setCurrentStream(stream);

            if (webcamVideoRef.current) {
                webcamVideoRef.current.srcObject = stream;
                showMessage(setRegisterMessage, 'Webcam started. Please position your face within the frame.', 'info'); // Simplified message

                webcamVideoRef.current.onloadedmetadata = async () => {
                    try {
                        await handlePlayVideo(webcamVideoRef.current);
                        // Optional: Add a small delay here if you still face issues with video frames not being ready
                        // await new Promise(resolve => setTimeout(resolve, 100));

                        const displaySize = {
                            width: webcamVideoRef.current.offsetWidth,
                            height: webcamVideoRef.current.offsetHeight
                        };
                        faceapi.matchDimensions(detectionCanvasRef.current, displaySize);
                        // The useEffect with dependencies will handle starting detection
                    } catch (err) {
                        console.error('onloadedmetadata: Error in handler:', err);
                        showMessage(setRegisterMessage, 'Error starting webcam. Please try again.', 'error');
                    }
                };
            }

        } catch (err) {
            console.error('startWebcam: Error accessing webcam:', err);
            showMessage(setRegisterMessage, 'Could not start webcam. Please check permissions.', 'error');
        }
    };

    const stopWebcam = () => {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            setCurrentStream(null);
        }

        if (faceDetectionIntervalRef.current) {
            clearInterval(faceDetectionIntervalRef.current);
            faceDetectionIntervalRef.current = null;
        }

        if (countdownIntervalRef.current) {
            clearInterval(countdownIntervalRef.current);
            countdownIntervalRef.current = null;
        }

        countdownActiveRef.current = false;
        faceAlignedRef.current = false;

        // Optional: Clear UI state if needed
        clearMessage(setRegisterMessage);
        clearMessage(setRegisterMessage);
        setCapturedImageDataURL(null);
        setFaceDescriptor(null);
        latestDescriptorRef.current = null;
        livenessVerifiedRef.current = false;
        lastSpokenTimeRef.current = 0;
        setIsPhotoAlreadyCaptured(false);
    };

    const capturePhoto = () => {

        if (!currentStream) {
            console.log('Capture failed: No current stream');
            showMessage(setRegisterMessage, 'Webcam not active.', 'error');
            return;
        }

        if (!faceAlignedRef.current) {
            console.log('Capture failed: Face not aligned at capture moment');
            showMessage(setRegisterMessage, 'Face moved out of alignment before capture. Please re-center.', 'warning');
            return;
        }

        try {
            const videoWidth = webcamVideoRef.current.videoWidth;
            const videoHeight = webcamVideoRef.current.videoHeight;
            const displayWidth = webcamVideoRef.current.offsetWidth;
            const displayHeight = webcamVideoRef.current.offsetHeight;

            const frameRect = faceFrameRef.current.getBoundingClientRect();
            const videoRect = webcamVideoRef.current.getBoundingClientRect();

            const frameX_display = frameRect.left - videoRect.left;
            const frameY_display = frameRect.top - videoRect.top;
            const frameWidth_display = frameRect.width;
            const frameHeight_display = frameRect.height;

            const scaleX = videoWidth / displayWidth;
            const scaleY = videoHeight / displayHeight;

            const paddingPercentage = 0.1;

            const sourceX = Math.max(0, (frameX_display * scaleX) - (frameWidth_display * scaleX * paddingPercentage));
            const sourceY = Math.max(0, (frameY_display * scaleY) - (frameHeight_display * scaleY * paddingPercentage));
            const sourceWidth = Math.min(videoWidth - sourceX, (frameWidth_display * scaleX) * (1 + paddingPercentage * 2));
            const sourceHeight = Math.min(videoHeight - sourceY, (frameHeight_display * scaleY) * (1 + paddingPercentage * 2));

            photoCanvasRef.current.width = sourceWidth;
            photoCanvasRef.current.height = sourceHeight;
            const context = photoCanvasRef.current.getContext('2d');

            context.drawImage(
                webcamVideoRef.current,
                sourceX, sourceY, sourceWidth, sourceHeight,
                0, 0, photoCanvasRef.current.width, photoCanvasRef.current.height
            );

            const dataURL = photoCanvasRef.current.toDataURL('image/jpeg', 0.9);
            setCapturedImageDataURL(dataURL);
            setFaceDescriptor(latestDescriptorRef.current);
            if (capturedPhotoRef.current) capturedPhotoRef.current.src = dataURL;

            // --- VISIBILITY CONTROL ---
            webcamDisplayRef.current?.classList.add('hidden'); // Hide webcam
            capturedDisplayRef.current?.classList.remove('hidden'); // Show captured photo

            setIsPhotoAlreadyCaptured(true); // This state change is key for stopping detection
            faceAlignedRef.current = false;

            // Ensure all intervals are stopped after capture
            if (faceDetectionIntervalRef.current) {
                clearInterval(faceDetectionIntervalRef.current);
                faceDetectionIntervalRef.current = null;
            }
            if (countdownIntervalRef.current) {
                clearInterval(countdownIntervalRef.current);
                countdownIntervalRef.current = null;
                console.log('capturePhoto: Stopped countdown interval.');
            }
            countdownActiveRef.current = false;

            showMessage(setRegisterMessage, 'Face captured successfully! You can now register.', 'success');
        } catch (err) {
            console.error('Error capturing photo:', err);
            showMessage(setRegisterMessage, 'Error capturing photo. Please try again.', 'error');
        }
    };

    const retakePhoto = () => {
        clearMessage(setRegisterMessage);
        clearMessage(setRegisterMessage);
        setCapturedImageDataURL(null);
        setFaceDescriptor(null);
        latestDescriptorRef.current = null;
        livenessVerifiedRef.current = false;
        lastSpokenTimeRef.current = 0;
        setIsPhotoAlreadyCaptured(false); // This state change is key for restarting detection
        countdownActiveRef.current = false;

        // --- VISIBILITY CONTROL ---
        capturedDisplayRef.current?.classList.add('hidden'); // Hide captured photo
        webcamDisplayRef.current?.classList.remove('hidden'); // Show webcam

        startWebcam(); // This will re-initialize the webcam and its state, triggering useEffect to restart detection
    };

    const registerUser = async () => {
        setIsLoading(true);
        clearMessage(setRegisterMessage);

        if (!capturedImageDataURL) {
            setIsLoading(false);
            showMessage(setRegisterMessage, 'No captured face detected. Please retake or capture a new face.', 'error');
            return;
        }

        const formData = new FormData();

        const isLogin = type === "login";
        const apiEndpoint = isLogin ? "/login" : "/register";

        formData.append('employeeId', employeeId);
        formData.append('faceDescriptor', faceDescriptor);
        try {
            const response = await fetch(`${API_BASE_URL}${apiEndpoint}`, {
                method: 'POST',
                body: formData,
            });
            const data = await response.json();
            if (data.success) {
                setIsLoading(false);
                if (!isLogin) {
                    setAlert({
                        open: true,
                        type: 'success',
                        message: 'Face registered successfully!'
                    });
                    navigate('/dashboard/manageemployees');
                } else {
                    setLoginInfo(data);
                }
                onClose();
            } else {
                setIsLoading(false);
                setAlert({
                    open: true,
                    type: 'error',
                    message: `${isLogin ? 'Login' : 'Registration'} failed: ${data.detail || 'Unknown error.'}`
                });
            }
        } catch (error) {
            setIsLoading(false);
            setAlert({
                type: 'error',
                message: `${isLogin ? 'Login' : 'Registration'} failed due to network or server error.`
            });
        }
    };

    useEffect(() => {
        if (modelsLoaded && currentStream && webcamVideoRef.current && !isPhotoAlreadyCaptured) {
            // Clear any existing interval before setting a new one
            if (faceDetectionIntervalRef.current) {
                clearInterval(faceDetectionIntervalRef.current);
            }
            faceDetectionIntervalRef.current = setInterval(detectFaces, 150);
        } else {
            // Stop detection if conditions are not met (e.g., photo captured, stream stopped)
            if (faceDetectionIntervalRef.current) {
                clearInterval(faceDetectionIntervalRef.current);
                faceDetectionIntervalRef.current = null;
            }
            if (countdownIntervalRef.current) {
                clearInterval(countdownIntervalRef.current);
                countdownIntervalRef.current = null;
                countdownActiveRef.current = false;
            }
            // This is important: if detection stops because a photo was captured, reset faceAlignedRef
            if (isPhotoAlreadyCaptured) {
                faceAlignedRef.current = false;
            }
        }

        // Cleanup for this specific effect
        return () => {
            if (faceDetectionIntervalRef.current) {
                clearInterval(faceDetectionIntervalRef.current);
                faceDetectionIntervalRef.current = null;
            }
            if (countdownIntervalRef.current) {
                clearInterval(countdownIntervalRef.current);
                countdownIntervalRef.current = null;
            }
            countdownActiveRef.current = false;
        };
    }, [modelsLoaded, currentStream, isPhotoAlreadyCaptured]);

    useEffect(() => {
        if (open) {
            loadModels().then(startWebcam);
        } else {
            stopWebcam();
        }
    }, [open]);

    return (
        <React.Fragment>
            <BootstrapDialog
                open={open}
                // onClose={onClose}
                aria-labelledby="customized-dialog-title"
                fullScreen
            >
                <Components.DialogTitle sx={{ m: 0, p: 2, color: theme.palette.primary.text.main }} id="customized-dialog-title">
                    {
                        type === "login" ? "Login with Face" : "Register Face"
                    }
                </Components.DialogTitle>

                <Components.IconButton
                    aria-label="close"
                    onClick={onClose}
                    sx={(theme) => ({
                        position: 'absolute',
                        right: 8,
                        top: 8,
                        color: theme.palette.primary.icon,
                    })}
                >
                    <CustomIcons iconName={'fa-solid fa-xmark'} css='cursor-pointer text-black w-5 h-5' />
                </Components.IconButton>


                <Components.DialogContent dividers>
                    <div className='flex justify-center items-center'>
                        <div className="bg-white w-[300px] max-w-md flex flex-col gap-5 text-center">
                            <div
                                id="webcamDisplay"
                                ref={webcamDisplayRef}
                                className={`video-container relative rounded-lg overflow-hidden bg-black h-[28rem] ${isPhotoAlreadyCaptured ? 'hidden' : ''}`}
                            >
                                <video
                                    id="webcamVideo"
                                    ref={webcamVideoRef}
                                    autoPlay
                                    muted
                                    playsInline
                                    className="absolute top-0 left-0 w-full h-full object-cover"
                                />
                                <canvas
                                    id="detectionCanvas"
                                    ref={detectionCanvasRef}
                                    className="absolute top-0 left-0 w-full h-full"
                                />
                                <div
                                    className="face-frame absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-60 md:w-52 h-4/5 md:max-w-[25rem] max-h-[32rem] border-2 border-opacity-80 border-red-500 rounded-xl pointer-events-none z-10 transition-colors"
                                    ref={faceFrameRef}
                                >
                                    <div className="corner tl absolute top-[-2px] left-[-2px] w-5 h-5 border-4 border-blue-500 border-r-0 border-b-0 rounded-tl-lg"></div>
                                    <div className="corner tr absolute top-[-2px] right-[-2px] w-5 h-5 border-4 border-blue-500 border-l-0 border-b-0 rounded-tr-lg"></div>
                                    <div className="corner bl absolute bottom-[-2px] left-[-2px] w-5 h-5 border-4 border-blue-500 border-r-0 border-t-0 rounded-bl-lg"></div>
                                    <div className="corner br absolute bottom-[-2px] right-[-2px] w-5 h-5 border-4 border-blue-500 border-l-0 border-t-0 rounded-br-lg"></div>
                                </div>
                            </div>

                            <div
                                id="capturedDisplay"
                                ref={capturedDisplayRef}
                                className={`captured-display relative rounded-lg overflow-hidden shadow h-[28rem] w-full max-w-[500px] mx-auto ${!isPhotoAlreadyCaptured ? 'hidden' : ''
                                    }`}
                            >
                                <img
                                    id="capturedPhoto"
                                    ref={capturedPhotoRef}
                                    alt="Captured Preview"
                                    className=" w-full h-full object-cover rounded-lg"
                                />

                                {/* Retake Button Overlay */}
                                <div className='absolute bottom-5 right-2 z-50 bg-white bg-opacity-90 rounded-full shadow-lg cursor-pointer hover:scale-110 transition-transform'>
                                    <Components.IconButton
                                        id="retakeIcon"
                                        onClick={retakePhoto}
                                    >
                                        <CustomIcons iconName="fa-solid fa-rotate-left" css="text-blue-600 w-5 h-5" />
                                    </Components.IconButton>
                                </div>
                            </div>


                            <canvas id="photoCanvas" ref={photoCanvasRef} className="hidden"></canvas>
                            {registerMessage.text && (
                                <p className={`message rounded-lg px-3 py-2 text-sm my-2 ${registerMessage.type === 'success' ? 'bg-green-100 text-green-800' :
                                    registerMessage.type === 'error' ? 'bg-red-100 text-red-800' :
                                        registerMessage.type === 'info' ? 'bg-blue-100 text-blue-800' :
                                            'bg-yellow-100 text-yellow-800'
                                    }`}>
                                    {registerMessage.text}
                                </p>
                            )}
                        </div>
                    </div>
                </Components.DialogContent>

                <Components.DialogActions>
                    <div className='flex justify-end'>
                        <Button type={`button`} text={type === "login" ? "Login" : "Register Face"} isLoading={isLoading} onClick={() => registerUser()} />
                    </div>
                </Components.DialogActions>
            </BootstrapDialog>
        </React.Fragment>
    );
}

const mapDispatchToProps = {
    setAlert,
};

export default connect(null, mapDispatchToProps)(FaceRegistration);

// faceDescriptorInput.value [-0.20307719707489014,0.08489348739385605,0.09423123300075531,-0.04917548596858978,-0.011448781937360764,-0.09727434813976288,-0.04818405583500862,-0.0314621701836586,0.09653551876544952,-0.04819248989224434,0.2129346877336502,-0.08544760197401047,-0.18603187799453735,-0.07560791820287704,-0.04770626500248909,0.08892910182476044,-0.14932917058467865,-0.16694702208042145,-0.10572270303964615,-0.12741591036319733,0.05167757719755173,-0.03788996487855911,-0.04347049072384834,0.057397641241550446,-0.22920803725719452,-0.3402428925037384,-0.08062141388654709,-0.10273782908916473,0.01164971012622118,-0.13035665452480316,-0.046662259846925735,0.04977964609861374,-0.197633758187294,-0.05426344648003578,0.011346585117280483,0.0893159955739975,0.10185516625642776,0.011521969921886921,0.16923312842845917,0.00038316677091643214,-0.13229191303253174,0.01890481635928154,0.1187531054019928,0.3004947900772095,0.19400173425674438,0.13401690125465393,0.00931297056376934,0.026701023802161217,0.1396358758211136,-0.22519400715827942,0.12019090354442596,0.12482412159442902,0.14514024555683136,0.07571904361248016,0.11791591346263885,-0.19271455705165863,-0.008687866851687431,0.07900340855121613,-0.13239970803260803,0.12837688624858856,-0.0008295245352201164,-0.08938662707805634,-0.004948951303958893,-0.02468997985124588,0.22155222296714783,0.10682907700538635,-0.1205705851316452,-0.05273295193910599,0.16775883734226227,-0.11143025755882263,0.033432092517614365,0.04068842902779579,-0.0754178911447525,-0.121733658015728,-0.2565939724445343,0.1011243686079979,0.44103869795799255,0.142691507935524,-0.2005525827407837,0.048655252903699875,-0.07192613184452057,-0.05994788929820061,0.051672834903001785,0.024576961994171143,-0.18102702498435974,0.05892806127667427,-0.11263203620910645,0.06589040160179138,0.22066165506839752,0.1084270104765892,-0.018233541399240494,0.17953333258628845,-0.04684571176767349,0.039075903594493866,0.05055942386388779,0.026941604912281036,-0.1768411546945572,-0.04030369222164154,-0.05411892756819725,-0.015518746338784695,0.07569931447505951,-0.10832324624061584,-0.012202106416225433,0.08130376040935516,-0.16637150943279266,0.08600275218486786,-0.019971415400505066,0.017994349822402,-0.10247708112001419,0.12386839836835861,-0.05932646244764328,-0.019221697002649307,0.08295237272977829,-0.2375868558883667,0.15785646438598633,0.1580338180065155,-0.003705449402332306,0.16316203773021698,0.0937310978770256,0.032121170312166214,0.047959521412849426,0.018765168264508247,-0.05763811990618706,-0.0917239710688591,0.053658321499824524,-0.036152973771095276,0.14981886744499207,0.07738353312015533]


// imageBlob Blob {size: 136979, type: 'image/jpeg'}