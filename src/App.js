import React, { useEffect, useRef, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgl';
import { AutoModel, AutoProcessor, RawImage } from '@xenova/transformers';

const ObjectDetection = () => {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const [model, setModel] = useState(null);
    const [processor, setProcessor] = useState(null);

    useEffect(() => {
        const loadModelAndProcessor = async () => {
            const loadedModel = await AutoModel.from_pretrained('onnx-community/yolov10s');
            const loadedProcessor = await AutoProcessor.from_pretrained('onnx-community/yolov10s');
            setModel(loadedModel);
            setProcessor(loadedProcessor);
        };
        loadModelAndProcessor();
        setupCamera();
    }, []);

    const setupCamera = async () => {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
            videoRef.current.play();
            setupCanvas();
        };
    };

    const setupCanvas = () => {
        const videoWidth = videoRef.current.videoWidth;
        const videoHeight = videoRef.current.videoHeight;
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
        const ctx = canvasRef.current.getContext('2d');
        ctx.translate(videoWidth, 0);
        ctx.scale(-1, 1); // Mirror the canvas for a selfie view
    };

    const renderPrediction = async () => {
        const ctx = canvasRef.current.getContext('2d');
        ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

        const image = await RawImage.read(canvasRef.current.toDataURL());
        const { pixel_values, reshaped_input_sizes } = await processor(image);

        const { output0 } = await model({ images: pixel_values });
        const predictions = output0.tolist()[0];

        const threshold = 0.5;
        const [newHeight, newWidth] = reshaped_input_sizes[0];
        const [xs, ys] = [image.width / newWidth, image.height / newHeight];

        predictions.forEach(([xmin, ymin, xmax, ymax, score, id]) => {
            if (score >= threshold) {
                const bbox = [xmin * xs, ymin * ys, xmax * xs, ymax * ys].map(x => parseFloat(x.toFixed(2)));
                drawBoundingBox(ctx, bbox, model.config.id2label[id], score);
            }
        });

        requestAnimationFrame(renderPrediction);
    };

    const drawBoundingBox = (ctx, bbox, label, score) => {
        ctx.strokeStyle = 'red';
        ctx.lineWidth = 2;
        ctx.strokeRect(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
        ctx.fillStyle = 'red';
        ctx.font = '12px Arial';
        ctx.fillText(`${label} (${score.toFixed(2)})`, bbox[0], bbox[1] > 10 ? bbox[1] - 5 : 10);
    };

    return (
        <div>
            <video ref={videoRef} style={{ display: 'none' }} />
            <canvas ref={canvasRef} />
            {model && processor && renderPrediction()}
        </div>
    );
};

export default ObjectDetection;
