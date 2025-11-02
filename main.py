# main.py
import streamlit as st
import torch
import numpy as np
import os
from core.scene_generator import TextTo3DGenerator
from core.nerf_renderer import NeRFRenderer
from core.diffusion_model import Diffusion3DModel
from core.mesh_converter import MeshGenerator
from utils.visualization import SceneVisualizer
from utils.config import load_config
import tempfile

st.set_page_config(
    page_title="3D Scene Generator - Text to 3D Revolution - wasif",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="expanded"
)

def initialize_session_state():
    if 'scene_generator' not in st.session_state:
        st.session_state.scene_generator = None
    if 'nerf_renderer' not in st.session_state:
        st.session_state.nerf_renderer = None
    if 'generated_scenes' not in st.session_state:
        st.session_state.generated_scenes = {}
    if 'current_scene' not in st.session_state:
        st.session_state.current_scene = None

def load_components():
    with st.spinner("🔄 Loading 3D Scene Generation Engine..."):
        if st.session_state.scene_generator is None:
            st.session_state.scene_generator = TextTo3DGenerator()
        if st.session_state.nerf_renderer is None:
            st.session_state.nerf_renderer = NeRFRenderer()

def main():
    st.title("🔄 3D Scene Generator - Text to 3D Revolution")
    st.markdown("Generate complete 3D scenes and environments from text descriptions using neural radiance fields and diffusion models")
    
    initialize_session_state()
    
    with st.sidebar:
        st.header("⚙️ Generation Configuration")
        
        generation_mode = st.selectbox(
            "Generation Mode",
            ["Text to 3D Scene", "Text to 360° Environment", "Interactive Scene Editing", "Style Transfer"],
            help="Select the type of 3D generation"
        )
        
        scene_complexity = st.select_slider(
            "Scene Complexity",
            options=["Simple", "Medium", "Complex", "Highly Detailed"],
            value="Medium"
        )
        
        st.subheader("Model Parameters")
        model_architecture = st.selectbox(
            "Model Architecture",
            ["NeRF + Diffusion", "DreamFusion", "Magic3D", "Zero-1-to-3"]
        )
        
        enable_lighting = st.checkbox("Dynamic Lighting", value=True)
        enable_materials = st.checkbox("Material Generation", value=True)
        enable_physics = st.checkbox("Physics Simulation", value=False)
        
        st.subheader("Output Format")
        output_format = st.selectbox(
            "3D Format",
            ["NeRF", "Mesh (OBJ)", "Point Cloud", "Voxel Grid", "Gaussian Splatting"]
        )
        
        resolution = st.slider("Output Resolution", 64, 1024, 256, 64)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎨 Text Input", "🤖 Generation", "👁️ 3D Viewer", "🛠️ Editing", "🚀 Export"])
    
    with tab1:
        st.header("Scene Description")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            scene_description = st.text_area(
                "Describe your 3D scene",
                height=150,
                placeholder="A cozy living room with a large window, comfortable sofa, coffee table, and potted plants. Soft morning light streaming through the window..."
            )
            
            style_prompt = st.text_input(
                "Style Reference (optional)",
                placeholder="photorealistic, cinematic lighting, 8k resolution"
            )
        
        with col2:
            st.subheader("Scene Parameters")
            scene_scale = st.selectbox("Scene Scale", ["Small Object", "Room", "Building", "Landscape"])
            lighting_condition = st.selectbox("Lighting", ["Daylight", "Night", "Sunset", "Studio", "Moody"])
            camera_angle = st.selectbox("Camera Angle", ["Front", "Side", "Top", "45°", "Free"])
            
            st.subheader("Advanced")
            random_seed = st.number_input("Random Seed", value=42, min_value=0, max_value=1000000)
            guidance_scale = st.slider("Guidance Scale", 1.0, 20.0, 7.5, 0.5)
        
        if st.button("🎨 Generate 3D Scene", type="primary") and scene_description:
            generate_3d_scene(scene_description, style_prompt, scene_scale, generation_mode, output_format, resolution)
    
    with tab2:
        st.header("Generation Progress")
        
        if st.session_state.current_scene:
            display_generation_progress()
        else:
            st.info("🎨 Enter a scene description and click 'Generate 3D Scene' to start")
    
    with tab3:
        st.header("3D Scene Viewer")
        
        if st.session_state.current_scene:
            display_3d_viewer()
        else:
            st.info("👆 Generate a scene first to view it in 3D")
    
    with tab4:
        st.header("Scene Editing")
        
        if st.session_state.current_scene:
            display_editing_tools()
        else:
            st.info("Generate a scene to enable editing tools")
    
    with tab5:
        st.header("Export & Deployment")
        
        if st.session_state.current_scene:
            display_export_options()
        else:
            st.info("Generate a scene to export options")

def generate_3d_scene(description, style_prompt, scale, mode, output_format, resolution):
    load_components()
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("🎨 Processing text description...")
        progress_bar.progress(10)
        
        status_text.text("🤖 Initializing diffusion model...")
        progress_bar.progress(25)
        
        status_text.text("🔮 Generating 3D representation...")
        progress_bar.progress(50)
        
        scene_data = st.session_state.scene_generator.generate_scene(
            text_prompt=description,
            style_prompt=style_prompt,
            scene_scale=scale,
            output_format=output_format,
            resolution=resolution,
            guidance_scale=7.5
        )
        
        status_text.text("🎭 Rendering with NeRF...")
        progress_bar.progress(75)
        
        rendered_scene = st.session_state.nerf_renderer.render_scene(
            scene_data=scene_data,
            output_format=output_format
        )
        
        scene_id = f"scene_{len(st.session_state.generated_scenes) + 1}"
        st.session_state.generated_scenes[scene_id] = {
            'description': description,
            'scene_data': scene_data,
            'rendered_scene': rendered_scene,
            'metadata': {
                'scale': scale,
                'mode': mode,
                'format': output_format,
                'resolution': resolution
            }
        }
        st.session_state.current_scene = scene_id
        
        progress_bar.progress(100)
        status_text.text("✅ 3D scene generation completed!")
        
        st.balloons()
        
    except Exception as e:
        st.error(f"❌ Scene generation failed: {str(e)}")

def display_generation_progress():
    scene_id = st.session_state.current_scene
    scene_data = st.session_state.generated_scenes[scene_id]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Generation Details")
        st.write(f"**Description:** {scene_data['description']}")
        st.write(f"**Scale:** {scene_data['metadata']['scale']}")
        st.write(f"**Format:** {scene_data['metadata']['format']}")
        st.write(f"**Resolution:** {scene_data['metadata']['resolution']}px")
    
    with col2:
        st.subheader("Rendering Preview")
        
        if 'preview_images' in scene_data['rendered_scene']:
            previews = scene_data['rendered_scene']['preview_images']
            if len(previews) > 0:
                st.image(previews[0], caption="Generated View", use_column_width=True)

def display_3d_viewer():
    scene_id = st.session_state.current_scene
    scene_data = st.session_state.generated_scenes[scene_id]
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("Interactive 3D Viewer")
        
        visualizer = SceneVisualizer()
        plotly_fig = visualizer.create_interactive_viewer(scene_data['rendered_scene'])
        st.plotly_chart(plotly_fig, use_container_width=True)
    
    with col2:
        st.subheader("View Controls")
        
        camera_view = st.selectbox("Camera View", ["Perspective", "Top", "Front", "Side", "Free"])
        lighting = st.select_slider("Lighting", options=["Soft", "Medium", "Bright", "Dramatic"])
        background = st.selectbox("Background", ["Transparent", "White", "Black", "Studio", "Gradient"])
        
        if st.button("🔄 Reset Camera"):
            st.rerun()
        
        st.subheader("Scene Stats")
        if 'stats' in scene_data['rendered_scene']:
            stats = scene_data['rendered_scene']['stats']
            st.write(f"**Vertices:** {stats.get('vertices', 0):,}")
            st.write(f"**Faces:** {stats.get('faces', 0):,}")
            st.write(f"**Resolution:** {stats.get('resolution', 'N/A')}")

def display_editing_tools():
    scene_id = st.session_state.current_scene
    scene_data = st.session_state.generated_scenes[scene_id]
    
    st.subheader("Scene Editing Tools")
    
    tab1, tab2, tab3, tab4 = st.tabs(["✏️ Geometry", "🎨 Materials", "💡 Lighting", "🌍 Environment"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            scale_factor = st.slider("Scale Factor", 0.1, 3.0, 1.0, 0.1)
            smoothness = st.slider("Mesh Smoothness", 0.0, 1.0, 0.5, 0.1)
        
        with col2:
            if st.button("🔄 Apply Geometry Changes"):
                st.info("Geometry editing applied")
    
    with tab2:
        material_type = st.selectbox("Material Type", ["Diffuse", "Metallic", "Transparent", "Emissive"])
        base_color = st.color_picker("Base Color", "#808080")
        roughness = st.slider("Roughness", 0.0, 1.0, 0.5, 0.1)
        
        if st.button("🎨 Apply Materials"):
            st.info("Materials applied to scene")
    
    with tab3:
        light_type = st.selectbox("Light Type", ["Point", "Directional", "Spot", "Area"])
        light_intensity = st.slider("Light Intensity", 0.0, 5.0, 1.0, 0.1)
        light_color = st.color_picker("Light Color", "#FFFFFF")
        
        if st.button("💡 Update Lighting"):
            st.info("Lighting configuration updated")
    
    with tab4:
        environment_map = st.selectbox("Environment Map", ["None", "Studio", "Outdoor", "Night", "Sunset", "Custom"])
        fog_density = st.slider("Fog Density", 0.0, 1.0, 0.0, 0.1)
        
        if st.button("🌍 Apply Environment"):
            st.info("Environment settings applied")

def display_export_options():
    scene_id = st.session_state.current_scene
    scene_data = st.session_state.generated_scenes[scene_id]
    
    st.subheader("Export 3D Scene")
    
    export_format = st.selectbox(
        "Export Format",
        ["OBJ + MTL", "GLTF/GLB", "FBX", "PLY", "STL", "USDZ", "Blender File"]
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        include_textures = st.checkbox("Include Textures", value=True)
        include_materials = st.checkbox("Include Materials", value=True)
    
    with col2:
        export_animation = st.checkbox("Export Animation", value=False)
        export_lods = st.checkbox("Export LODs", value=False)
    
    with col3:
        compression = st.selectbox("Compression", ["None", "Zip", "High Compression"])
    
    if st.button("📥 Export Scene", type="primary"):
        with st.spinner("Exporting scene..."):
            try:
                mesh_generator = MeshGenerator()
                export_data = mesh_generator.export_scene(
                    scene_data['rendered_scene'],
                    format=export_format,
                    include_textures=include_textures,
                    include_materials=include_materials
                )
                
                st.success("✅ Scene exported successfully!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.download_button(
                        label="📥 Download 3D File",
                        data=export_data['file_data'],
                        file_name=export_data['filename'],
                        mime=export_data['mime_type']
                    )
                
                with col2:
                    if 'preview_image' in export_data:
                        st.image(export_data['preview_image'], caption="Export Preview", use_column_width=True)
            
            except Exception as e:
                st.error(f"❌ Export failed: {str(e)}")

if __name__ == "__main__":
    main()