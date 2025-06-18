# Hierarchical Progressive Reconstruction Pipeline
# Implementation of research-backed strategy for coherent multi-view reconstruction
# Author: Clinton T. Kunhardt

import os
import cv2
import torch
import numpy as np
import sys
import glob
import time
import open3d as o3d
from typing import Dict, List, Optional, Tuple, Set
import gc
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import json
import pickle
import networkx as nx
from collections import defaultdict
import itertools

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

class ImageCorrelationAnalyzer:
    """
    Analyzes correlation between images based on feature matches and spatial distribution
    """
    
    def __init__(self):
        self.sift = cv2.SIFT_create(nfeatures=1000)
        self.matcher = cv2.BFMatcher()
        
    def compute_image_correlation_matrix(self, image_paths: List[str]) -> np.ndarray:
        """
        Compute correlation matrix between all image pairs
        Returns normalized correlation scores [0,1]
        """
        print(f"Computing correlation matrix for {len(image_paths)} images...")
        n_images = len(image_paths)
        correlation_matrix = np.zeros((n_images, n_images))
        
        # Extract features for all images
        features = {}
        for i, path in enumerate(image_paths):
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                # Resize for faster processing
                h, w = img.shape
                max_dim = 800
                if max(h, w) > max_dim:
                    scale = max_dim / max(h, w)
                    img = cv2.resize(img, (int(w * scale), int(h * scale)))
                
                kp, desc = self.sift.detectAndCompute(img, None)
                features[i] = (kp, desc)
            else:
                features[i] = ([], None)
        
        # Compute pairwise correlations
        for i in range(n_images):
            correlation_matrix[i, i] = 1.0  # Self-correlation
            
            for j in range(i + 1, n_images):
                corr_score = self._compute_pair_correlation(features[i], features[j])
                correlation_matrix[i, j] = corr_score
                correlation_matrix[j, i] = corr_score
                
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{n_images} images")
        
        return correlation_matrix
    
    def _compute_pair_correlation(self, feat1: Tuple, feat2: Tuple) -> float:
        """
        Compute correlation between two image features
        Considers both number and spatial distribution of matches
        """
        kp1, desc1 = feat1
        kp2, desc2 = feat2
        
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        # Find matches
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return 0.0
        
        # Number of matches component (normalized)
        match_score = min(len(good_matches) / 50.0, 1.0)
        
        # Spatial distribution component
        points1 = np.array([kp1[m.queryIdx].pt for m in good_matches])
        points2 = np.array([kp2[m.trainIdx].pt for m in good_matches])
        
        # Compute spatial spread (higher is better for coverage)
        spread1 = np.std(points1, axis=0).mean()
        spread2 = np.std(points2, axis=0).mean()
        spread_score = min((spread1 + spread2) / 100.0, 1.0)
        
        # Combined score
        correlation_score = 0.7 * match_score + 0.3 * spread_score
        
        return correlation_score

class HierarchicalClusterer:
    """
    Progressive clustering with dynamic adjustment for weakly associated regions
    """
    
    def __init__(self, min_cluster_size: int = 8, max_cluster_size: int = 16, overlap_ratio: float = 0.4):
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.overlap_ratio = overlap_ratio
        
    def create_hierarchical_clusters(self, image_paths: List[str], correlation_matrix: np.ndarray) -> List[Dict]:
        """
        Create hierarchical clusters with quality evaluation and dynamic adjustment
        Goal: Create spatial/geometric clusters, not chronological ones
        """
        print("Creating hierarchical clusters...")
        
        # Step 1: Analyze correlation matrix for natural groupings (building sides)
        geometric_clusters = self._create_geometric_clusters(image_paths, correlation_matrix)
        
        # Step 2: Add overlap clusters to bridge between building sides
        bridge_clusters = self._create_bridge_clusters(geometric_clusters, correlation_matrix)
        
        # Step 3: Ensure complete coverage with gap-filling clusters
        coverage_clusters = self._ensure_complete_coverage(image_paths, geometric_clusters + bridge_clusters, correlation_matrix)
        
        # Step 4: Optimize and evaluate
        all_clusters = geometric_clusters + bridge_clusters + coverage_clusters
        final_clusters = self._optimize_and_evaluate_clusters(all_clusters, correlation_matrix)
        
        print(f"Created {len(final_clusters)} spatially-aware clusters")
        return final_clusters
    
    def _create_geometric_clusters(self, image_paths: List[str], corr_matrix: np.ndarray) -> List[Dict]:
        """
        Create clusters based on geometric/spatial relationships using correlation analysis
        """
        n_images = len(image_paths)
        
        # Find strongly connected components (building sides/viewpoints)
        # Use higher threshold to find clear geometric groupings
        high_corr_threshold = 0.4  # Lower than before to catch more relationships
        adjacency_matrix = (corr_matrix > high_corr_threshold).astype(int)
        
        # Use networkx to find connected components
        G = nx.from_numpy_array(adjacency_matrix)
        connected_components = list(nx.connected_components(G))
        
        # Filter and process components
        geometric_clusters = []
        cluster_id = 0
        
        print(f"Found {len(connected_components)} connected components with correlation > {high_corr_threshold}")
        
        for component in connected_components:
            component_list = list(component)
            
            if len(component) >= self.min_cluster_size:
                # Large components might represent entire building sides
                if len(component) <= self.max_cluster_size:
                    # Perfect size - keep as is
                    cluster_dict = {
                        'core_images': component,
                        'anchor_images': set(),
                        'local_images': set(),
                        'cluster_id': cluster_id,
                        'cluster_type': 'geometric',
                        'all_images': component_list
                    }
                    geometric_clusters.append(cluster_dict)
                    cluster_id += 1
                else:
                    # Large component - split using internal correlation
                    sub_clusters = self._split_large_component(component_list, corr_matrix)
                    for sub_cluster in sub_clusters:
                        if len(sub_cluster) >= self.min_cluster_size:
                            cluster_dict = {
                                'core_images': set(sub_cluster),
                                'anchor_images': set(),
                                'local_images': set(),
                                'cluster_id': cluster_id,
                                'cluster_type': 'geometric_split',
                                'all_images': sub_cluster
                            }
                            geometric_clusters.append(cluster_dict)
                            cluster_id += 1
        
        print(f"Created {len(geometric_clusters)} geometric clusters")
        return geometric_clusters
    
    def _split_large_component(self, component_list: List[int], corr_matrix: np.ndarray) -> List[List[int]]:
        """
        Split large component using k-means clustering on correlation features
        """
        if len(component_list) <= self.max_cluster_size:
            return [component_list]
        
        # Extract correlation features for this component
        component_correlations = corr_matrix[np.ix_(component_list, component_list)]
        
        # Use k-means to split
        n_clusters = (len(component_list) + self.max_cluster_size - 1) // self.max_cluster_size
        n_clusters = min(n_clusters, len(component_list) // self.min_cluster_size)
        
        if n_clusters <= 1:
            return [component_list]
        
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(component_correlations)
        
        # Group by cluster labels
        sub_clusters = []
        for cluster_idx in range(n_clusters):
            sub_cluster_indices = [component_list[i] for i, label in enumerate(cluster_labels) if label == cluster_idx]
            if len(sub_cluster_indices) >= self.min_cluster_size:
                sub_clusters.append(sub_cluster_indices)
        
        return sub_clusters
    
    def _create_bridge_clusters(self, geometric_clusters: List[Dict], corr_matrix: np.ndarray) -> List[Dict]:
        """
        Create bridge clusters that connect different geometric regions
        These help with registration between building sides
        """
        bridge_clusters = []
        cluster_id = 2000  # High ID to distinguish from geometric clusters
        
        # Find images that have moderate correlation with multiple geometric clusters
        all_geometric_images = set()
        for cluster in geometric_clusters:
            all_geometric_images.update(cluster['all_images'])
        
        n_images = corr_matrix.shape[0]
        unassigned_images = set(range(n_images)) - all_geometric_images
        
        if not unassigned_images:
            return bridge_clusters
        
        # For each unassigned image, check correlation with geometric clusters
        bridge_candidates = []
        
        for img_idx in unassigned_images:
            cluster_correlations = []
            for cluster in geometric_clusters:
                cluster_images = list(cluster['all_images'])
                correlations = [corr_matrix[img_idx, cluster_img] for cluster_img in cluster_images]
                avg_correlation = np.mean(correlations)
                cluster_correlations.append((cluster['cluster_id'], avg_correlation))
            
            # Sort by correlation
            cluster_correlations.sort(key=lambda x: x[1], reverse=True)
            
            # If this image correlates reasonably with multiple clusters, it's a bridge candidate
            if len(cluster_correlations) >= 2 and cluster_correlations[1][1] > 0.2:
                bridge_candidates.append({
                    'image_idx': img_idx,
                    'primary_cluster': cluster_correlations[0][0],
                    'secondary_cluster': cluster_correlations[1][0],
                    'primary_corr': cluster_correlations[0][1],
                    'secondary_corr': cluster_correlations[1][1]
                })
        
        # Group bridge candidates into bridge clusters
        if bridge_candidates:
            # Group by primary-secondary cluster pairs
            pair_groups = defaultdict(list)
            for candidate in bridge_candidates:
                pair_key = tuple(sorted([candidate['primary_cluster'], candidate['secondary_cluster']]))
                pair_groups[pair_key].append(candidate['image_idx'])
            
            # Create bridge clusters
            for pair_key, bridge_images in pair_groups.items():
                if len(bridge_images) >= self.min_cluster_size // 2:  # Smaller bridges are OK
                    # Add some images from the connected geometric clusters for context
                    extended_bridge = list(bridge_images)
                    
                    # Add a few images from each connected cluster
                    for cluster in geometric_clusters:
                        if cluster['cluster_id'] in pair_key:
                            cluster_images = list(cluster['all_images'])[:3]  # Add first 3 images
                            extended_bridge.extend(cluster_images)
                    
                    if len(extended_bridge) >= self.min_cluster_size:
                        cluster_dict = {
                            'core_images': set(bridge_images),
                            'anchor_images': set(extended_bridge) - set(bridge_images),
                            'local_images': set(),
                            'cluster_id': cluster_id,
                            'cluster_type': 'bridge',
                            'all_images': extended_bridge,
                            'connects_clusters': list(pair_key)
                        }
                        bridge_clusters.append(cluster_dict)
                        cluster_id += 1
        
        print(f"Created {len(bridge_clusters)} bridge clusters")
        return bridge_clusters
    
    def _ensure_complete_coverage(self, image_paths: List[str], existing_clusters: List[Dict], corr_matrix: np.ndarray) -> List[Dict]:
        """
        Ensure all images are covered - create gap-filling clusters for orphaned images
        """
        n_images = len(image_paths)
        covered_images = set()
        
        for cluster in existing_clusters:
            covered_images.update(cluster['all_images'])
        
        uncovered_images = set(range(n_images)) - covered_images
        
        coverage_clusters = []
        cluster_id = 3000
        
        if uncovered_images:
            print(f"Found {len(uncovered_images)} uncovered images, creating coverage clusters")
            
            # Group uncovered images by proximity in the sequence (last resort)
            uncovered_list = sorted(list(uncovered_images))
            
            for i in range(0, len(uncovered_list), self.max_cluster_size):
                cluster_images = uncovered_list[i:i + self.max_cluster_size]
                
                if len(cluster_images) >= self.min_cluster_size:
                    cluster_dict = {
                        'core_images': set(cluster_images),
                        'anchor_images': set(),
                        'local_images': set(),
                        'cluster_id': cluster_id,
                        'cluster_type': 'coverage',
                        'all_images': cluster_images
                    }
                    coverage_clusters.append(cluster_dict)
                    cluster_id += 1
                elif len(cluster_images) > 0:
                    # Small group - try to merge with nearest existing cluster
                    best_cluster = None
                    best_avg_corr = 0
                    
                    for existing_cluster in existing_clusters:
                        existing_images = list(existing_cluster['all_images'])
                        correlations = []
                        for orphan_img in cluster_images:
                            for existing_img in existing_images[:5]:  # Check correlation with first 5
                                correlations.append(corr_matrix[orphan_img, existing_img])
                        
                        avg_corr = np.mean(correlations) if correlations else 0
                        if avg_corr > best_avg_corr:
                            best_avg_corr = avg_corr
                            best_cluster = existing_cluster
                    
                    # Add to best cluster if correlation is reasonable
                    if best_cluster and best_avg_corr > 0.15:
                        best_cluster['local_images'].update(cluster_images)
                        best_cluster['all_images'].extend(cluster_images)
        
        return coverage_clusters
    
    def _optimize_and_evaluate_clusters(self, all_clusters: List[Dict], corr_matrix: np.ndarray) -> List[Dict]:
        """
        Optimize clusters and evaluate quality with relaxed thresholds
        """
        optimized_clusters = []
        
        for cluster in all_clusters:
            all_images = set(cluster['all_images'])
            
            # Compute quality score
            quality_score = self._compute_cluster_quality_score(all_images, corr_matrix)
            cluster['quality_score'] = quality_score
            
            # Very relaxed quality thresholds based on cluster type
            if cluster['cluster_type'] == 'geometric':
                min_quality = 0.25  # Geometric clusters should be good
            elif cluster['cluster_type'] == 'bridge':
                min_quality = 0.15  # Bridge clusters can be lower quality
            else:
                min_quality = 0.1   # Coverage clusters just need to exist
            
            if quality_score > min_quality and len(all_images) >= self.min_cluster_size:
                optimized_clusters.append(cluster)
            else:
                print(f"Filtered cluster {cluster['cluster_id']} ({cluster['cluster_type']}) - quality: {quality_score:.3f}")
        
        # Ensure minimum number of clusters
        if len(optimized_clusters) < 8:
            print("Too few clusters, adding back filtered ones...")
            all_clusters.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
            optimized_clusters = all_clusters[:max(8, len(optimized_clusters))]
        
        return optimized_clusters
    
    def _identify_seed_clusters(self, image_paths: List[str], corr_matrix: np.ndarray) -> List[Set[int]]:
        """
        Identify seed clusters using correlation-based clustering
        """
        n_images = len(image_paths)
        
        # Convert correlation to distance
        distance_matrix = 1.0 - corr_matrix
        
        # Determine optimal number of clusters
        n_clusters = max(3, min(8, n_images // self.max_cluster_size))
        
        # Use spectral clustering for better handling of non-convex clusters
        from sklearn.cluster import SpectralClustering
        clustering = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', random_state=42)
        cluster_labels = clustering.fit_predict(corr_matrix)
        
        # Convert to sets
        seed_clusters = []
        for cluster_id in range(n_clusters):
            cluster_indices = set(np.where(cluster_labels == cluster_id)[0])
            if len(cluster_indices) >= self.min_cluster_size:
                seed_clusters.append(cluster_indices)
        
        return seed_clusters
    
    def _expand_clusters_with_overlap(self, seed_clusters: List[Set[int]], corr_matrix: np.ndarray) -> List[Dict]:
        """
        Expand clusters to include overlapping images for continuity
        """
        expanded_clusters = []
        n_images = corr_matrix.shape[0]
        
        for i, seed_cluster in enumerate(seed_clusters):
            cluster_dict = {
                'core_images': seed_cluster,
                'anchor_images': set(),
                'local_images': set(),
                'cluster_id': i
            }
            
            # Find anchor images (high correlation with multiple clusters)
            potential_anchors = set()
            for img_idx in range(n_images):
                if img_idx not in seed_cluster:
                    # Check correlation with this cluster
                    cluster_correlations = [corr_matrix[img_idx, core_idx] for core_idx in seed_cluster]
                    avg_correlation = np.mean(cluster_correlations)
                    
                    if avg_correlation > 0.3:  # Threshold for anchor candidacy
                        potential_anchors.add(img_idx)
            
            # Select best anchors
            if potential_anchors:
                anchor_scores = []
                for anchor_idx in potential_anchors:
                    # Score based on correlation with core images
                    core_correlations = [corr_matrix[anchor_idx, core_idx] for core_idx in seed_cluster]
                    score = np.mean(core_correlations)
                    anchor_scores.append((anchor_idx, score))
                
                # Take top anchors
                anchor_scores.sort(key=lambda x: x[1], reverse=True)
                n_anchors = min(4, len(anchor_scores))
                cluster_dict['anchor_images'] = set([idx for idx, _ in anchor_scores[:n_anchors]])
            
            # Add local images to reach target size
            remaining_images = set(range(n_images)) - seed_cluster - cluster_dict['anchor_images']
            if remaining_images:
                local_scores = []
                for local_idx in remaining_images:
                    core_correlations = [corr_matrix[local_idx, core_idx] for core_idx in seed_cluster]
                    score = np.mean(core_correlations)
                    local_scores.append((local_idx, score))
                
                local_scores.sort(key=lambda x: x[1], reverse=True)
                target_local_size = self.max_cluster_size - len(seed_cluster) - len(cluster_dict['anchor_images'])
                target_local_size = max(0, target_local_size)
                
                cluster_dict['local_images'] = set([idx for idx, _ in local_scores[:target_local_size]])
            
            expanded_clusters.append(cluster_dict)
        
        return expanded_clusters
    
    def _dynamic_adjustment(self, clusters: List[Dict], corr_matrix: np.ndarray) -> List[Dict]:
        """
        Dynamic adjustment for weakly associated regions
        """
        adjusted_clusters = []
        
        for cluster in clusters:
            all_images = cluster['core_images'] | cluster['anchor_images'] | cluster['local_images']
            
            # Compute internal cluster correlation
            internal_correlations = []
            for i, j in itertools.combinations(all_images, 2):
                internal_correlations.append(corr_matrix[i, j])
            
            avg_internal_corr = np.mean(internal_correlations) if internal_correlations else 0.0
            
            # If cluster is weakly associated, try to improve it
            if avg_internal_corr < 0.4:
                # Remove weakest local images
                if cluster['local_images']:
                    local_scores = []
                    for local_idx in cluster['local_images']:
                        core_correlations = [corr_matrix[local_idx, core_idx] for core_idx in cluster['core_images']]
                        score = np.mean(core_correlations)
                        local_scores.append((local_idx, score))
                    
                    # Keep only top half of local images
                    local_scores.sort(key=lambda x: x[1], reverse=True)
                    keep_count = len(local_scores) // 2
                    cluster['local_images'] = set([idx for idx, _ in local_scores[:keep_count]])
            
            adjusted_clusters.append(cluster)
        
        return adjusted_clusters
    
    def _evaluate_cluster_quality(self, clusters: List[Dict], corr_matrix: np.ndarray) -> List[Dict]:
        """
        Evaluate cluster quality and filter poor reconstructions
        """
        quality_clusters = []
        
        for cluster in clusters:
            all_images = cluster['core_images'] | cluster['anchor_images'] | cluster['local_images']
            
            # Quality metrics
            quality_score = self._compute_cluster_quality_score(all_images, corr_matrix)
            
            cluster['quality_score'] = quality_score
            cluster['all_images'] = list(all_images)
            
            # Filter based on quality threshold
            if quality_score > 0.3 and len(all_images) >= self.min_cluster_size:
                quality_clusters.append(cluster)
            else:
                print(f"Filtered out cluster {cluster['cluster_id']} (quality: {quality_score:.3f})")
        
        return quality_clusters
    
    def _compute_cluster_quality_score(self, image_indices: Set[int], corr_matrix: np.ndarray) -> float:
        """
        Compute quality score for a cluster based on internal coherence
        """
        if len(image_indices) < 2:
            return 0.0
        
        # Internal correlation score
        internal_correlations = []
        for i, j in itertools.combinations(image_indices, 2):
            internal_correlations.append(corr_matrix[i, j])
        
        internal_score = np.mean(internal_correlations)
        
        # Size penalty (too small or too large clusters are penalized)
        size_penalty = 1.0
        if len(image_indices) < self.min_cluster_size:
            size_penalty = len(image_indices) / self.min_cluster_size
        elif len(image_indices) > self.max_cluster_size:
            size_penalty = self.max_cluster_size / len(image_indices)
        
        return internal_score * size_penalty

class RobustFeatureRegistration:
    """
    FPFH feature extraction and RANSAC-based alignment
    """
    
    def __init__(self, voxel_size: float = 0.05):
        self.voxel_size = voxel_size
        
    def extract_fpfh_features(self, point_cloud: o3d.geometry.PointCloud) -> np.ndarray:
        """
        Extract 33-dimensional FPFH features from point cloud
        """
        # Downsample for feature extraction
        pcd_down = point_cloud.voxel_down_sample(self.voxel_size)
        
        # Estimate normals
        pcd_down.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
        )
        
        # Compute FPFH features
        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 5, max_nn=100)
        )
        
        return pcd_down, fpfh
    
    def ransac_registration(self, source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud,
                          source_fpfh: o3d.pipelines.registration.Feature, target_fpfh: o3d.pipelines.registration.Feature) -> np.ndarray:
        """
        RANSAC-based registration with geometric constraints
        """
        distance_threshold = self.voxel_size * 1.5
        
        # RANSAC registration
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_pcd, target_pcd, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
        )
        
        return result.transformation
    
    def icp_refinement(self, source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud,
                      initial_transform: np.ndarray) -> np.ndarray:
        """
        ICP refinement for sub-pixel accuracy
        """
        distance_threshold = self.voxel_size * 0.4
        
        # Ensure target point cloud has normals for point-to-plane ICP
        if not target_pcd.has_normals():
            target_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
            )
        
        # Ensure source point cloud has normals too
        if not source_pcd.has_normals():
            source_pcd.estimate_normals(
                search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=self.voxel_size * 2, max_nn=30)
            )
        
        result = o3d.pipelines.registration.registration_icp(
            source_pcd, target_pcd, distance_threshold, initial_transform,
            o3d.pipelines.registration.TransformationEstimationPointToPlane()
        )
        
        return result.transformation

class BundleAdjustmentOptimizer:
    """
    Global bundle adjustment for joint optimization of cameras and structure
    """
    
    def __init__(self):
        self.convergence_threshold = 1e-6
        self.max_iterations = 100
        
    def global_bundle_adjustment(self, batch_predictions: List[Dict], anchor_correspondences: Dict) -> List[Dict]:
        """
        Joint optimization of all camera poses and 3D structure
        """
        print("Performing global bundle adjustment...")
        
        # Collect all cameras and points
        all_cameras = []
        all_points = []
        all_observations = []
        
        batch_offset = 0
        for batch_idx, batch in enumerate(batch_predictions):
            # Extract cameras
            cameras = batch['extrinsic']  # (N, 3, 4)
            for cam_idx, camera in enumerate(cameras):
                all_cameras.append({
                    'batch_idx': batch_idx,
                    'local_idx': cam_idx,
                    'global_idx': len(all_cameras),
                    'extrinsic': camera,
                    'intrinsic': batch['intrinsic'][cam_idx]
                })
            
            # Extract 3D points
            points_3d = batch['world_points_from_depth']  # (N, H, W, 3)
            for cam_idx in range(points_3d.shape[0]):
                points = points_3d[cam_idx].reshape(-1, 3)
                valid_mask = ~np.any(np.isnan(points) | np.isinf(points), axis=1)
                valid_points = points[valid_mask]
                
                for pt_idx, point in enumerate(valid_points):
                    all_points.append({
                        'batch_idx': batch_idx,
                        'cam_idx': cam_idx,
                        'point_3d': point,
                        'global_idx': len(all_points)
                    })
            
            batch_offset += len(cameras)
        
        # Optimize using scipy minimize (simplified version)
        optimized_predictions = self._optimize_cameras_and_structure(batch_predictions, anchor_correspondences)
        
        return optimized_predictions
    
    def _optimize_cameras_and_structure(self, batch_predictions: List[Dict], anchor_correspondences: Dict) -> List[Dict]:
        """
        Simplified optimization focusing on camera alignment
        """
        if len(batch_predictions) <= 1:
            return batch_predictions
        
        # Use first batch as reference
        reference_batch = batch_predictions[0]
        optimized_batches = [reference_batch]
        
        for i, batch in enumerate(batch_predictions[1:], 1):
            print(f"Optimizing batch {i+1}/{len(batch_predictions)}...")
            
            # Find anchor point correspondences
            ref_anchors, batch_anchors = self._find_anchor_correspondences(
                reference_batch, batch, anchor_correspondences
            )
            
            if len(ref_anchors) >= 4:
                # Compute robust transformation
                transform = self._compute_robust_transformation(ref_anchors, batch_anchors)
                
                # Apply transformation to batch
                optimized_batch = self._apply_transformation_to_batch(batch, transform)
                optimized_batches.append(optimized_batch)
                
                print(f"Batch {i+1} aligned using {len(ref_anchors)} anchor correspondences")
            else:
                print(f"Warning: Insufficient anchors for batch {i+1}, using centroid alignment")
                # Fallback alignment
                optimized_batch = self._centroid_alignment(reference_batch, batch)
                optimized_batches.append(optimized_batch)
        
        return optimized_batches
    
    def _find_anchor_correspondences(self, ref_batch: Dict, batch: Dict, anchor_correspondences: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find 3D point correspondences between batches using anchor images
        """
        ref_anchors = []
        batch_anchors = []
        
        # Use image paths to find correspondences
        ref_paths = ref_batch.get('image_paths', [])
        batch_paths = batch.get('image_paths', [])
        
        common_images = set(ref_paths) & set(batch_paths)
        
        for common_img in list(common_images)[:5]:  # Limit to avoid too many correspondences
            if common_img in ref_paths and common_img in batch_paths:
                ref_idx = ref_paths.index(common_img)
                batch_idx = batch_paths.index(common_img)
                
                # Get 3D points for this image
                ref_points = ref_batch['world_points_from_depth'][ref_idx]
                batch_points = batch['world_points_from_depth'][batch_idx]
                
                # Compute centroids as anchor points
                ref_centroid = np.mean(ref_points.reshape(-1, 3), axis=0)
                batch_centroid = np.mean(batch_points.reshape(-1, 3), axis=0)
                
                # Filter out invalid points
                if not (np.any(np.isnan(ref_centroid)) or np.any(np.isnan(batch_centroid))):
                    ref_anchors.append(ref_centroid)
                    batch_anchors.append(batch_centroid)
        
        return np.array(ref_anchors), np.array(batch_anchors)
    
    def _compute_robust_transformation(self, ref_points: np.ndarray, batch_points: np.ndarray) -> np.ndarray:
        """
        Compute robust transformation using RANSAC
        """
        if len(ref_points) < 3:
            return np.eye(4)
        
        # Use Open3D's robust registration
        ref_pcd = o3d.geometry.PointCloud()
        ref_pcd.points = o3d.utility.Vector3dVector(ref_points)
        
        batch_pcd = o3d.geometry.PointCloud()
        batch_pcd.points = o3d.utility.Vector3dVector(batch_points)
        
        # Estimate transformation
        threshold = 0.5  # meters
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            batch_pcd, ref_pcd,
            o3d.utility.Vector2iVector([(i, i) for i in range(len(ref_points))]),
            threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(threshold)],
            o3d.pipelines.registration.RANSACConvergenceCriteria(4000, 0.999)
        )
        
        return result.transformation
    
    def _apply_transformation_to_batch(self, batch: Dict, transform: np.ndarray) -> Dict:
        """
        Apply transformation to all 3D content in a batch
        """
        transformed_batch = batch.copy()
        
        # Transform world points
        world_points = batch['world_points_from_depth']
        original_shape = world_points.shape
        
        # Reshape and transform
        points_flat = world_points.reshape(-1, 3)
        valid_mask = ~np.any(np.isnan(points_flat) | np.isinf(points_flat), axis=1)
        
        # Apply transformation to valid points
        points_homo = np.ones((points_flat.shape[0], 4))
        points_homo[:, :3] = points_flat
        points_homo[~valid_mask] = [0, 0, 0, 1]  # Keep invalid points as is
        
        transformed_points = (transform @ points_homo.T).T
        transformed_points = transformed_points[:, :3]
        
        # Restore invalid points
        transformed_points[~valid_mask] = points_flat[~valid_mask]
        
        transformed_batch['world_points_from_depth'] = transformed_points.reshape(original_shape)
        
        return transformed_batch
    
    def _centroid_alignment(self, ref_batch: Dict, batch: Dict) -> Dict:
        """
        Fallback centroid-based alignment
        """
        ref_points = ref_batch['world_points_from_depth'].reshape(-1, 3)
        batch_points = batch['world_points_from_depth'].reshape(-1, 3)
        
        # Filter valid points
        ref_valid = ~np.any(np.isnan(ref_points) | np.isinf(ref_points), axis=1)
        batch_valid = ~np.any(np.isnan(batch_points) | np.isinf(batch_points), axis=1)
        
        if np.sum(ref_valid) > 0 and np.sum(batch_valid) > 0:
            ref_centroid = np.mean(ref_points[ref_valid], axis=0)
            batch_centroid = np.mean(batch_points[batch_valid], axis=0)
            
            translation = ref_centroid - batch_centroid
            
            # Create translation transform
            transform = np.eye(4)
            transform[:3, 3] = translation
            
            return self._apply_transformation_to_batch(batch, transform)
        
        return batch

class HierarchicalReconstructionPipeline:
    """
    Main pipeline implementing the complete hierarchical reconstruction strategy
    """
    
    def __init__(self, max_resolution: int = 384, voxel_size: float = 0.05):
        self.max_resolution = max_resolution
        self.voxel_size = voxel_size
        self.device = device
        self.model = None
        
        # Initialize components with more permissive settings
        self.correlation_analyzer = ImageCorrelationAnalyzer()
        self.clusterer = HierarchicalClusterer(
            min_cluster_size=8,    # Smaller minimum
            max_cluster_size=16,   # Keep reasonable maximum
            overlap_ratio=0.4      # More overlap for better continuity
        )
        self.feature_registration = RobustFeatureRegistration(voxel_size)
        self.bundle_adjuster = BundleAdjustmentOptimizer()
        
    def load_vggt_model(self):
        """Load VGGT model"""
        if self.model is not None:
            return
            
        print("Loading VGGT model...")
        start_time = time.time()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model = VGGT()
        _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        
        state_dict = torch.hub.load_state_dict_from_url(_URL, map_location='cpu')
        self.model.load_state_dict(state_dict)
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
    
    def process_large_dataset(self, target_dir: str) -> Dict:
        """
        Complete hierarchical reconstruction pipeline
        """
        print("="*60)
        print("HIERARCHICAL PROGRESSIVE RECONSTRUCTION PIPELINE")
        print("="*60)
        
        # Find all images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext)))
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext.upper())))
        
        image_paths = sorted(image_paths)
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            raise ValueError("No images found")
        
        # Load VGGT model
        self.load_vggt_model()
        
        # Phase 1: Image Correlation Analysis
        print("\n" + "="*40)
        print("PHASE 1: IMAGE CORRELATION ANALYSIS")
        print("="*40)
        
        correlation_matrix = self.correlation_analyzer.compute_image_correlation_matrix(image_paths)
        
        # Save correlation matrix for analysis
        np.save(os.path.join(target_dir, "correlation_matrix.npy"), correlation_matrix)
        
        # Phase 2: Hierarchical Clustering
        print("\n" + "="*40)
        print("PHASE 2: HIERARCHICAL CLUSTERING")
        print("="*40)
        
        clusters = self.clusterer.create_hierarchical_clusters(image_paths, correlation_matrix)
        
        # Save cluster information
        cluster_info = []
        for cluster in clusters:
            cluster_paths = [image_paths[i] for i in cluster['all_images']]
            cluster_info.append({
                'cluster_id': cluster['cluster_id'],
                'quality_score': cluster['quality_score'],
                'image_count': len(cluster['all_images']),
                'image_paths': cluster_paths
            })
        
        with open(os.path.join(target_dir, "cluster_info.json"), 'w') as f:
            json.dump(cluster_info, f, indent=2)
        
        # Phase 3: VGGT Reconstruction per Cluster
        print("\n" + "="*40)
        print("PHASE 3: VGGT RECONSTRUCTION PER CLUSTER")
        print("="*40)
        
        batch_predictions = []
        for i, cluster in enumerate(clusters):
            cluster_image_paths = [image_paths[idx] for idx in cluster['all_images']]
            print(f"\nProcessing cluster {i+1}/{len(clusters)}: {len(cluster_image_paths)} images")
            
            try:
                prediction = self._process_cluster(cluster_image_paths, cluster['cluster_id'])
                prediction['cluster_info'] = cluster
                batch_predictions.append(prediction)
            except Exception as e:
                print(f"Failed to process cluster {i+1}: {e}")
                continue
        
        if not batch_predictions:
            raise RuntimeError("All clusters failed to process")
        
        # Phase 4: Feature-Based Registration
        print("\n" + "="*40)
        print("PHASE 4: ROBUST FEATURE REGISTRATION")
        print("="*40)
        
        registered_predictions = self._register_clusters(batch_predictions)
        
        # Phase 5: Global Bundle Adjustment
        print("\n" + "="*40)
        print("PHASE 5: GLOBAL BUNDLE ADJUSTMENT")
        print("="*40)
        
        # Create anchor correspondences from cluster overlaps
        anchor_correspondences = self._build_anchor_correspondences(clusters, image_paths)
        
        final_predictions = self.bundle_adjuster.global_bundle_adjustment(
            registered_predictions, anchor_correspondences
        )
        
        # Phase 6: Merge Results
        print("\n" + "="*40)
        print("PHASE 6: MERGING FINAL RESULTS")
        print("="*40)
        
        merged_prediction = self._merge_predictions(final_predictions)
        
        print("\n" + "="*60)
        print("RECONSTRUCTION COMPLETE!")
        print("="*60)
        
        return merged_prediction
    
    def _process_cluster(self, image_paths: List[str], cluster_id: int) -> Dict:
        """
        Process a single cluster with VGGT
        """
        # Preprocess images
        processed_paths = self._preprocess_image_paths(image_paths)
        
        try:
            # Load and preprocess
            images = load_and_preprocess_images(processed_paths).to(self.device)
            
            # Run inference
            with torch.no_grad():
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
                with torch.cuda.amp.autocast(dtype=dtype):
                    predictions = self.model(images)
            
            # Process predictions
            extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
            predictions["extrinsic"] = extrinsic
            predictions["intrinsic"] = intrinsic
            
            # Convert to numpy
            for key in predictions.keys():
                if isinstance(predictions[key], torch.Tensor):
                    predictions[key] = predictions[key].cpu().numpy().squeeze(0)
            
            # Generate world points
            depth_map = predictions["depth"]
            world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
            predictions["world_points_from_depth"] = world_points
            
            # Add metadata
            predictions["image_paths"] = image_paths
            predictions["cluster_id"] = cluster_id
            
            return predictions
            
        finally:
            # Cleanup
            for path in processed_paths:
                if path.startswith("temp_resized_") and os.path.exists(path):
                    os.remove(path)
            torch.cuda.empty_cache()
            gc.collect()
    
    def _preprocess_image_paths(self, image_paths: List[str]) -> List[str]:
        """
        Preprocess images (resize if needed)
        """
        processed_paths = []
        
        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            h, w = img.shape[:2]
            max_dim = max(h, w)
            
            if max_dim > self.max_resolution:
                scale = self.max_resolution / max_dim
                new_w, new_h = int(w * scale), int(h * scale)
                img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                base_name = os.path.basename(img_path)
                temp_path = f"temp_resized_{base_name}"
                cv2.imwrite(temp_path, img_resized)
                processed_paths.append(temp_path)
            else:
                processed_paths.append(img_path)
        
        return processed_paths
    
    def _register_clusters(self, batch_predictions: List[Dict]) -> List[Dict]:
        """
        Register clusters using FPFH features and RANSAC
        """
        if len(batch_predictions) <= 1:
            return batch_predictions
        
        registered_predictions = [batch_predictions[0]]  # Reference
        
        for i, batch in enumerate(batch_predictions[1:], 1):
            print(f"Registering cluster {i+1} to reference...")
            
            # Convert to point clouds
            ref_points = registered_predictions[0]['world_points_from_depth'].reshape(-1, 3)
            batch_points = batch['world_points_from_depth'].reshape(-1, 3)
            
            # Filter valid points
            ref_valid = ~np.any(np.isnan(ref_points) | np.isinf(ref_points), axis=1)
            batch_valid = ~np.any(np.isnan(batch_points) | np.isinf(batch_points), axis=1)
            
            if np.sum(ref_valid) < 100 or np.sum(batch_valid) < 100:
                print(f"Insufficient valid points for registration, using centroid alignment")
                registered_batch = self.bundle_adjuster._centroid_alignment(registered_predictions[0], batch)
                registered_predictions.append(registered_batch)
                continue
            
            # Create point clouds
            ref_pcd = o3d.geometry.PointCloud()
            ref_pcd.points = o3d.utility.Vector3dVector(ref_points[ref_valid])
            
            batch_pcd = o3d.geometry.PointCloud()
            batch_pcd.points = o3d.utility.Vector3dVector(batch_points[batch_valid])
            
            try:
                # Extract FPFH features
                ref_pcd_down, ref_fpfh = self.feature_registration.extract_fpfh_features(ref_pcd)
                batch_pcd_down, batch_fpfh = self.feature_registration.extract_fpfh_features(batch_pcd)
                
                # RANSAC registration
                ransac_transform = self.feature_registration.ransac_registration(
                    batch_pcd_down, ref_pcd_down, batch_fpfh, ref_fpfh
                )
                
                # ICP refinement
                final_transform = self.feature_registration.icp_refinement(
                    batch_pcd, ref_pcd, ransac_transform
                )
                
                # Apply transformation
                registered_batch = self.bundle_adjuster._apply_transformation_to_batch(batch, final_transform)
                registered_predictions.append(registered_batch)
                
                print(f"Cluster {i+1} successfully registered")
                
            except Exception as e:
                print(f"Registration failed for cluster {i+1}: {e}, using centroid alignment")
                registered_batch = self.bundle_adjuster._centroid_alignment(registered_predictions[0], batch)
                registered_predictions.append(registered_batch)
        
        return registered_predictions
    
    def _build_anchor_correspondences(self, clusters: List[Dict], image_paths: List[str]) -> Dict:
        """
        Build robust anchor correspondences between clusters using better overlap detection
        """
        anchor_correspondences = {}
        
        # Method 1: Direct image overlap detection
        image_to_clusters = defaultdict(list)
        
        for cluster in clusters:
            for img_idx in cluster['all_images']:
                if img_idx < len(image_paths):  # Safety check
                    img_path = image_paths[img_idx]
                    image_to_clusters[img_path].append(cluster['cluster_id'])
        
        # Find images that appear in multiple clusters
        for img_path, cluster_ids in image_to_clusters.items():
            if len(cluster_ids) > 1:
                anchor_correspondences[img_path] = cluster_ids
        
        # Method 2: Find bridge clusters that explicitly connect other clusters
        bridge_connections = {}
        for cluster in clusters:
            if cluster.get('cluster_type') == 'bridge' and 'connects_clusters' in cluster:
                connected_clusters = cluster['connects_clusters']
                for img_idx in cluster['core_images']:
                    if img_idx < len(image_paths):
                        img_path = image_paths[img_idx]
                        if img_path not in anchor_correspondences:
                            anchor_correspondences[img_path] = connected_clusters
                        else:
                            # Extend existing connections
                            existing_connections = set(anchor_correspondences[img_path])
                            new_connections = set(connected_clusters)
                            anchor_correspondences[img_path] = list(existing_connections | new_connections)
        
        # Method 3: Add high-correlation image pairs as potential anchors
        correlation_anchors = 0
        for i, cluster_a in enumerate(clusters):
            for j, cluster_b in enumerate(clusters[i+1:], i+1):
                # Find best correlated image pairs between clusters
                cluster_a_images = [idx for idx in cluster_a['all_images'] if idx < len(image_paths)]
                cluster_b_images = [idx for idx in cluster_b['all_images'] if idx < len(image_paths)]
                
                if cluster_a_images and cluster_b_images:
                    # Find the best matching pair
                    best_corr = 0
                    best_pair = None
                    
                    # Sample a few images from each cluster to avoid O(n²) computation
                    sample_a = cluster_a_images[:min(5, len(cluster_a_images))]
                    sample_b = cluster_b_images[:min(5, len(cluster_b_images))]
                    
                    for img_a in sample_a:
                        for img_b in sample_b:
                            # Use filename similarity as a proxy for spatial proximity
                            path_a = os.path.basename(image_paths[img_a])
                            path_b = os.path.basename(image_paths[img_b])
                            
                            # Extract numbers from filenames to check if they're sequential
                            try:
                                import re
                                num_a = [int(x) for x in re.findall(r'\d+', path_a)]
                                num_b = [int(x) for x in re.findall(r'\d+', path_b)]
                                
                                if num_a and num_b:
                                    # Check if images are close in sequence (potential viewpoint continuity)
                                    min_diff = min(abs(a - b) for a in num_a for b in num_b)
                                    if min_diff <= 3:  # Images within 3 steps might be related
                                        similarity_score = 1.0 / (1.0 + min_diff)
                                        if similarity_score > best_corr:
                                            best_corr = similarity_score
                                            best_pair = (img_a, img_b)
                            except:
                                pass
                    
                    # Add best pair as anchor if good enough
                    if best_pair and best_corr > 0.3:
                        img_a, img_b = best_pair
                        path_a = image_paths[img_a]
                        path_b = image_paths[img_b]
                        
                        # Add both images as anchors connecting these clusters
                        for path, img_idx in [(path_a, img_a), (path_b, img_b)]:
                            if path not in anchor_correspondences:
                                anchor_correspondences[path] = [cluster_a['cluster_id'], cluster_b['cluster_id']]
                                correlation_anchors += 1
        
        print(f"Found {len(anchor_correspondences)} anchor images ({correlation_anchors} from correlation analysis)")
        
        # Debug: Print some anchor info
        if len(anchor_correspondences) > 0:
            print("Sample anchor correspondences:")
            for i, (img_path, cluster_ids) in enumerate(list(anchor_correspondences.items())[:3]):
                img_name = os.path.basename(img_path)
                print(f"  {img_name} -> clusters {cluster_ids}")
        
        return anchor_correspondences
    
    def _merge_predictions(self, predictions_list: List[Dict]) -> Dict:
        """
        Merge multiple predictions into single coherent result
        """
        if len(predictions_list) == 1:
            return predictions_list[0]
        
        merged = {}
        
        # Merge arrays by concatenation
        array_keys = ['world_points_from_depth', 'depth', 'images', 'extrinsic', 'intrinsic']
        
        for key in array_keys:
            if key in predictions_list[0]:
                arrays = [pred[key] for pred in predictions_list]
                merged[key] = np.concatenate(arrays, axis=0)
        
        # Merge lists
        list_keys = ['image_paths']
        for key in list_keys:
            if key in predictions_list[0]:
                merged[key] = []
                for pred in predictions_list:
                    merged[key].extend(pred.get(key, []))
        
        # Add metadata
        merged['num_clusters'] = len(predictions_list)
        merged['cluster_sizes'] = [len(pred.get('image_paths', [])) for pred in predictions_list]
        
        return merged

def main():
    """
    Run the complete hierarchical reconstruction pipeline
    """
    target_dir = "C:\\repos\\gatech\\photogrammetry\\south-building"
    
    # Create pipeline
    pipeline = HierarchicalReconstructionPipeline(
        max_resolution=384,  # Conservative for GPU memory
        voxel_size=0.05     # Balance between detail and speed
    )
    
    try:
        # Run complete pipeline
        start_time = time.time()
        predictions = pipeline.process_large_dataset(target_dir)
        total_time = time.time() - start_time
        
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        
        # Save GLB
        glb_path = os.path.join(target_dir, f"hierarchical_reconstruction_{str(time.time())}.glb")
        print(f"Saving GLB to {glb_path}")
        scene = predictions_to_glb(predictions, conf_thres=50.0, target_dir=target_dir)
        scene.export(glb_path)
        
        # Visualize result
        world_points = predictions["world_points_from_depth"].reshape(-1, 3)
        valid_mask = ~np.any(np.isnan(world_points) | np.isinf(world_points), axis=1)
        valid_points = world_points[valid_mask]
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_points)
        
        print(f"\n--- FINAL RESULTS ---")
        print(f"Processed {len(predictions.get('image_paths', []))} images")
        print(f"Generated {len(valid_points)} valid 3D points")
        print(f"Used {predictions.get('num_clusters', 0)} clusters")
        print(f"Cluster sizes: {predictions.get('cluster_sizes', [])}")
        
        print("\nVisualizing hierarchical reconstruction...")
        o3d.visualization.draw_geometries([pcd])
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()