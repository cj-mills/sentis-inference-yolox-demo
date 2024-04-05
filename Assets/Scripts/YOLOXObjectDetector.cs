using System.Linq;
using UnityEngine;
using System;
using System.Collections.Generic;
using Unity.Sentis;

using CJM.YOLOXUtils;
using CJM.BBox2DToolkit;

namespace CJM.SentisInference.YOLOX
{
    /// <summary>
    /// YOLOXObjectDetector is a class that extends BarracudaModelRunner for object detection using the YOLOX model.
    /// It handles model execution, processes the output, and generates bounding boxes with corresponding labels and colors.
    /// The class supports various worker types, including PixelShader and ComputePrecompiled, as well as Async GPU Readback.
    /// </summary>
    public class YOLOXObjectDetector : SentisModelRunner
    {
        // Output Processing configuration and variables
        [Header("Output Processing")]
        // JSON file containing the color map for bounding boxes
        [SerializeField, Tooltip("JSON file with bounding box colormaps")]
        private TextAsset colormapFile;

        [Header("Settings")]
        [Tooltip("Interval (in frames) for unloading unused assets with Pixel Shader backend")]
        [SerializeField] private int pixelShaderUnloadInterval = 100;


        private int frameCounter = 0;

        // Stride values used by the YOLOX model
        private static readonly int[] Strides = { 8, 16, 32 };

        // Number of fields in each bounding box
        private const int NumBBoxFields = 5;

        // Layer names for the Transpose, Flatten, and TransposeOutput operations
        private string defaultOutputLayer;

        // Serializable classes to store color map information from JSON
        [System.Serializable]
        class Colormap
        {
            public string label;
            public List<float> color;
        }

        [System.Serializable]
        class ColormapList
        {
            public List<Colormap> items;
        }

        // List to store label and color pairs for each class
        private List<(string, Color)> colormapList = new List<(string, Color)>();

        // List to store grid and stride information for the YOLOX model
        private List<GridCoordinateAndStride> gridCoordsAndStrides = new List<GridCoordinateAndStride>();

        // Length of the proposal array for YOLOX output
        private int proposalLength;

        // Called at the start of the script
        protected override void Start()
        {
            base.Start();
            LoadColorMapList(); // Load colormap information from JSON file

            proposalLength = colormapList.Count + NumBBoxFields; // Calculate proposal length
        }

        // Load and prepare the YOLOX model
        protected override void LoadAndPrepareModel()
        {
            base.LoadAndPrepareModel();

            defaultOutputLayer = runtimeModel.outputs[0];

            // Set worker type for WebGL
            if (Application.platform == RuntimePlatform.WebGLPlayer)
            {
                backendType = BackendType.GPUPixel;
            }
        }

        /// <summary>
        /// Initialize the Barracuda engine
        /// <summary>
        protected override void InitializeEngine()
        {
            base.InitializeEngine();
        }

        /// <summary>
        /// Load the color map list from the JSON file
        /// <summary>
        private void LoadColorMapList()
        {
            if (IsColorMapListJsonNullOrEmpty())
            {
                Debug.LogError("Class labels JSON is null or empty.");
                return;
            }

            ColormapList colormapObj = DeserializeColorMapList(colormapFile.text);
            UpdateColorMap(colormapObj);
        }

        /// <summary>
        /// Check if the color map JSON file is null or empty
        /// <summary>
        private bool IsColorMapListJsonNullOrEmpty()
        {
            return colormapFile == null || string.IsNullOrWhiteSpace(colormapFile.text);
        }

        /// <summary>
        /// Deserialize the color map list from the JSON string
        /// <summary>
        private ColormapList DeserializeColorMapList(string json)
        {
            try
            {
                return JsonUtility.FromJson<ColormapList>(json);
            }
            catch (Exception ex)
            {
                Debug.LogError($"Failed to deserialize class labels JSON: {ex.Message}");
                return null;
            }
        }

        /// <summary>
        /// Update the color map list with deserialized data
        /// <summary>
        private void UpdateColorMap(ColormapList colormapObj)
        {
            if (colormapObj == null)
            {
                return;
            }

            // Add label and color pairs to the colormap list
            foreach (Colormap colormap in colormapObj.items)
            {
                Color color = new Color(colormap.color[0], colormap.color[1], colormap.color[2]);
                colormapList.Add((colormap.label, color));
            }
        }

        /// <summary>
        /// Execute the YOLOX model with the given input texture.
        /// </summary>
        public void ExecuteModel(RenderTexture inputTexture)
        {
            using (TensorFloat input = TextureConverter.ToTensor(inputTexture, channels: 3) as TensorFloat)
            {
                base.ExecuteModel(input);
            }

            // Update grid_strides if necessary
            if (engine.PeekOutput(defaultOutputLayer).shape.length / proposalLength != gridCoordsAndStrides.Count)
            {
                gridCoordsAndStrides = YOLOXUtility.GenerateGridCoordinatesWithStrides(Strides, inputTexture.height, inputTexture.width);
            }
        }

        /// <summary>
        /// Process the output array from the YOLOX model, applying Non-Maximum Suppression (NMS) and
        /// returning an array of BBox2DInfo objects with class labels and colors.
        /// </summary>
        /// <param name="outputArray">The output array from the YOLOX model</param>
        /// <param name="confidenceThreshold">The minimum confidence score for a bounding box to be considered</param>
        /// <param name="nms_threshold">The threshold for Non-Maximum Suppression (NMS)</param>
        /// <returns>An array of BBox2DInfo objects containing the filtered bounding boxes, class labels, and colors</returns>
        public BBox2DInfo[] ProcessOutput(float[] outputArray, float confidenceThreshold = 0.5f, float nms_threshold = 0.45f)
        {
            // Generate bounding box proposals from the output array
            List<BBox2D> proposals = YOLOXUtility.GenerateBoundingBoxProposals(outputArray, gridCoordsAndStrides, colormapList.Count, NumBBoxFields, confidenceThreshold);

            // Apply Non-Maximum Suppression (NMS) to the proposals
            List<int> proposal_indices = BBox2DUtility.NMSSortedBoxes(proposals, nms_threshold);

            // Create an array of BBox2DInfo objects containing the filtered bounding boxes, class labels, and colors
            return proposal_indices
                .Select(index => proposals[index])
                .Select(bbox => new BBox2DInfo(bbox, colormapList[bbox.index].Item1, colormapList[bbox.index].Item2))
                .ToArray();
        }

        /// <summary>
        /// Copy the model output to a float array.
        /// </summary>
        public float[] CopyOutputToArray()
        {
            using (TensorFloat output = engine.PeekOutput(defaultOutputLayer) as TensorFloat)
            {
                if (backendType == BackendType.GPUPixel)
                {
                    frameCounter++;
                    if (frameCounter % pixelShaderUnloadInterval == 0)
                    {
                        Resources.UnloadUnusedAssets();
                        frameCounter = 0;
                    }
                }
                output.TakeOwnership();
                output.MakeReadable();
                return output.ToReadOnlyArray();
            }
        }

        /// <summary>
        /// Crop input dimensions to be divisible by the maximum stride.
        /// </summary>
        public Vector2Int CropInputDims(Vector2Int inputDims)
        {
            inputDims[0] -= inputDims[0] % Strides.Max();
            inputDims[1] -= inputDims[1] % Strides.Max();

            return inputDims;
        }

        /// <summary>
        /// Clean up resources when the script is disabled.
        /// </summary>
        protected override void OnDisable()
        {
            base.OnDisable();
        }

    }
}