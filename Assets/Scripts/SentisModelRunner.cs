using UnityEngine;
using Unity.Sentis;
using System.Collections.Generic;

namespace CJM.SentisInference
{
    public abstract class SentisModelRunner : MonoBehaviour
    {
        [Header("Model Assets")]
        [SerializeField] protected ModelAsset model;

        [Tooltip("Execution device for the model")]
        [SerializeField] protected Unity.Sentis.DeviceType deviceType = Unity.Sentis.DeviceType.GPU;
        
        protected BackendType backendType;

        protected Model runtimeModel;
        protected IWorker engine;

        protected virtual void Start()
        {
            backendType = WorkerFactory.GetBestTypeForDevice(deviceType);
            Debug.Log($"Backend Type: {backendType}");
            LoadAndPrepareModel();
            InitializeEngine();
        }

        /// <summary>
        /// Load and prepare the model for execution.
        /// Override this method to apply custom modifications to the model.
        /// </summary>
        protected virtual void LoadAndPrepareModel()
        {
            runtimeModel = ModelLoader.Load(model);
        }

        
        /// <summary>
        /// Initialize the inference engine.
        /// </summary>
        protected virtual void InitializeEngine()
        {
            engine = WorkerFactory.CreateWorker(backendType, runtimeModel);
        }

        /// <summary>
        /// Execute the model with the given input Tensor.
        /// Override this method to implement custom input and output processing.
        /// </summary>
        /// <param name="input">The input Tensor for the model.</param>
        public virtual void ExecuteModel(TensorFloat input)
        {
            engine.Execute(input);
        }

        /// <summary>
        /// Execute the model with the given input Tensor.
        /// Override this method to implement custom input and output processing.
        /// </summary>
        /// <param name="input">The input Tensor for the model.</param>
        public virtual void ExecuteModel(IDictionary<string, Tensor> inputs)
        {
            engine.Execute(inputs);
        }

        /// <summary>
        /// Clean up resources when the component is disabled.
        /// </summary>
        protected virtual void OnDisable()
        {
            engine.Dispose();
        }
    }
}
