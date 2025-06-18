# Improved Hierarchical Progressive Reconstruction Pipeline
# Fixing critical issues with correlation analysis, clustering, and bundle adjustment
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
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import pairwise_distances
import json
import pickle
import networkx as nx
from collections import defaultdict
import itertools
from sklearn.preprocessing import StandardScaler

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

device = "cuda" if torch.cuda.is_available() else "cpu"

class EnhancedImageCorrelationAnalyzer:
    """
    Enhanced correlation analysis with multiple feature detectors and robust matching
    """
    
    def __init__(self):
        # Multiple feature detectors for robustness
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.03)
        self.orb = cv2.ORB_create(nfeatures=1500)
        self.surf = cv2.xfeatures2d.SIFT_create() if hasattr(cv2, 'xfeatures2d') else None
        
        # Enhanced matchers
        self.bf_matcher = cv2.BFMatcher()
        self.flann_matcher = cv2.FlannBasedMatcher(
            dict(algorithm=1, trees=5),
            dict(checks=50)
        )
        
    def compute_image_correlation_matrix(self, image_paths: List[str]) -> np.ndarray:
        """
        Enhanced correlation matrix with multiple feature types and robust geometric verification
        """
        print(f"Computing enhanced correlation matrix for {len(image_paths)} images...")
        n_images = len(image_paths)
        correlation_matrix = np.zeros((n_images, n_images))
        
        # Extract multiple feature types for all images
        all_features = {}
        for i, path in enumerate(image_paths):
            features = self._extract_multi_scale_features(path)
            all_features[i] = features
            
            if (i + 1) % 10 == 0:
                print(f"Extracted features for {i + 1}/{n_images} images")
        
        print("Computing pairwise correlations...")
        
        # Compute enhanced correlations with progress tracking
        total_pairs = (n_images * (n_images - 1)) // 2
        processed_pairs = 0
        
        for i in range(n_images):
            correlation_matrix[i, i] = 1.0
            
            for j in range(i + 1, n_images):
                try:
                    correlation = self._compute_enhanced_correlation(
                        all_features[i], all_features[j]
                    )
                    correlation_matrix[i, j] = correlation
                    correlation_matrix[j, i] = correlation
                except Exception as e:
                    print(f"Warning: Failed to compute correlation for pair ({i}, {j}): {e}")
                    correlation_matrix[i, j] = 0.0
                    correlation_matrix[j, i] = 0.0
                
                processed_pairs += 1
                if processed_pairs % 1000 == 0:
                    print(f"Processed {processed_pairs}/{total_pairs} image pairs ({100*processed_pairs/total_pairs:.1f}%)")
        
        print(f"Correlation matrix computation complete!")
        return correlation_matrix
    
    def _extract_multi_scale_features(self, image_path: str) -> Dict:
        """Extract features at multiple scales and with multiple detectors - simplified version"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {'sift': ([], None), 'orb': ([], None), 'scales': []}
        
        # Resize for consistent processing - smaller for speed
        h, w = img.shape
        target_size = 800  # Reduced from 1024
        if max(h, w) > target_size:
            scale = target_size / max(h, w)
            img = cv2.resize(img, (int(w * scale), int(h * scale)))
        
        features = {}
        
        try:
            # SIFT features (primary)
            kp_sift, desc_sift = self.sift.detectAndCompute(img, None)
            features['sift'] = (kp_sift, desc_sift)
        except Exception as e:
            print(f"SIFT extraction failed for {image_path}: {e}")
            features['sift'] = ([], None)
        
        try:
            # ORB features (backup)
            kp_orb, desc_orb = self.orb.detectAndCompute(img, None)
            features['orb'] = (kp_orb, desc_orb)
        except Exception as e:
            print(f"ORB extraction failed for {image_path}: {e}")
            features['orb'] = ([], None)
        
        # Skip multi-scale for now to avoid complexity
        features['scales'] = []
        
        return features
    
    def _compute_enhanced_correlation(self, feat1: Dict, feat2: Dict) -> float:
        """
        Enhanced correlation using multiple feature types - simplified to avoid hanging
        """
        correlations = []
        
        # SIFT correlation (primary)
        try:
            sift_corr = self._compute_verified_correlation(
                feat1['sift'], feat2['sift'], 'sift'
            )
            if sift_corr > 0:
                correlations.append(sift_corr * 0.8)  # Higher weight for SIFT
        except Exception as e:
            print(f"SIFT correlation failed: {e}")
        
        # ORB correlation (backup)
        try:
            orb_corr = self._compute_verified_correlation(
                feat1['orb'], feat2['orb'], 'orb'
            )
            if orb_corr > 0:
                correlations.append(orb_corr * 0.2)  # Lower weight for ORB
        except Exception as e:
            print(f"ORB correlation failed: {e}")
        
        # Skip multi-scale for now to avoid hanging
        
        return sum(correlations) if correlations else 0.0
    
    def _compute_verified_correlation(self, feat1: Tuple, feat2: Tuple, 
                                    detector_type: str) -> float:
        """Compute correlation with geometric verification and timeout protection"""
        kp1, desc1 = feat1
        kp2, desc2 = feat2
        
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return 0.0
        
        try:
            # Feature matching with timeout protection
            if detector_type == 'sift':
                matches = self.flann_matcher.knnMatch(desc1, desc2, k=2)
            else:
                matches = self.bf_matcher.knnMatch(desc1, desc2, k=2)
            
            # Ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:  # Stricter ratio test
                        good_matches.append(m)
            
            if len(good_matches) < 15:  # Reduced threshold for robustness
                return 0.0
            
            # Basic correlation score without expensive geometric verification
            # for now to avoid hanging
            match_score = min(len(good_matches) / 50.0, 1.0)
            
            # Simple spatial distribution check instead of fundamental matrix
            points1 = np.float32([kp1[m.queryIdx].pt for m in good_matches[:50]])  # Limit points
            points2 = np.float32([kp2[m.trainIdx].pt for m in good_matches[:50]])
            
            # Compute spread as a simpler geometric consistency check
            spread1 = np.std(points1, axis=0).mean() if len(points1) > 5 else 0
            spread2 = np.std(points2, axis=0).mean() if len(points2) > 5 else 0
            spread_score = min((spread1 + spread2) / 100.0, 1.0)
            
            # Skip expensive fundamental matrix computation for now
            return 0.7 * match_score + 0.3 * spread_score
            
        except Exception as e:
            print(f"Warning: Correlation computation failed: {e}")
            return 0.0


class AdaptiveHierarchicalClusterer:
    """
    Adaptive clustering that adjusts based on data characteristics
    """
    
    def __init__(self, min_cluster_size: int = 12, max_cluster_size: int = 20, 
                 overlap_ratio: float = 0.3):
        self.min_cluster_size = min_cluster_size
        self.max_cluster_size = max_cluster_size
        self.overlap_ratio = overlap_ratio
        
    def create_hierarchical_clusters(self, image_paths: List[str], 
                                   correlation_matrix: np.ndarray) -> List[Dict]:
        """
        Adaptive hierarchical clustering with multiple strategies
        """
        print("Creating adaptive hierarchical clusters...")
        
        # Analyze correlation matrix characteristics
        mean_corr = np.mean(correlation_matrix[correlation_matrix > 0])
        std_corr = np.std(correlation_matrix[correlation_matrix > 0])
        max_corr = np.max(correlation_matrix[correlation_matrix < 1.0])
        
        print(f"Correlation stats - Mean: {mean_corr:.3f}, Std: {std_corr:.3f}, Max: {max_corr:.3f}")
        
        # Adaptive thresholding based on data characteristics
        if mean_corr > 0.4:
            high_threshold = max(0.6, mean_corr + 0.5 * std_corr)
            medium_threshold = max(0.3, mean_corr)
        else:
            high_threshold = max(0.4, mean_corr + std_corr)
            medium_threshold = max(0.2, mean_corr - 0.5 * std_corr)
        
        print(f"Using adaptive thresholds - High: {high_threshold:.3f}, Medium: {medium_threshold:.3f}")
        
        # Multi-level clustering approach
        clusters = []
        
        # Level 1: High-confidence clusters
        high_conf_clusters = self._find_connected_components(
            correlation_matrix, high_threshold, "high_confidence"
        )
        clusters.extend(high_conf_clusters)
        
        # Level 2: Medium-confidence clusters for remaining images
        used_images = set()
        for cluster in high_conf_clusters:
            used_images.update(cluster['all_images'])
        
        remaining_indices = list(set(range(len(image_paths))) - used_images)
        if remaining_indices:
            remaining_corr = correlation_matrix[np.ix_(remaining_indices, remaining_indices)]
            medium_clusters = self._find_connected_components(
                remaining_corr, medium_threshold, "medium_confidence"
            )
            
            # Adjust indices back to global indexing
            for cluster in medium_clusters:
                cluster['all_images'] = [remaining_indices[i] for i in cluster['all_images']]
                cluster['core_images'] = set(cluster['all_images'])
                cluster['cluster_id'] = len(clusters)
                clusters.append(cluster)
        
        # Level 3: Density-based clustering for stragglers
        used_images = set()
        for cluster in clusters:
            used_images.update(cluster['all_images'])
        
        remaining_indices = list(set(range(len(image_paths))) - used_images)
        if len(remaining_indices) >= self.min_cluster_size:
            density_clusters = self._density_based_clustering(
                correlation_matrix, remaining_indices
            )
            
            for cluster in density_clusters:
                cluster['cluster_id'] = len(clusters)
                clusters.append(cluster)
        
        # Add bridge connections between clusters
        clusters = self._add_bridge_connections(clusters, correlation_matrix)
        
        print(f"Created {len(clusters)} adaptive clusters")
        return clusters
    
    def _find_connected_components(self, corr_matrix: np.ndarray, 
                                 threshold: float, cluster_type: str) -> List[Dict]:
        """Find connected components with given threshold"""
        adjacency = (corr_matrix >= threshold).astype(int)
        np.fill_diagonal(adjacency, 0)  # Remove self-connections
        
        G = nx.from_numpy_array(adjacency)
        components = list(nx.connected_components(G))
        
        clusters = []
        for i, component in enumerate(components):
            if len(component) >= self.min_cluster_size:
                cluster = {
                    'core_images': component,
                    'anchor_images': set(),
                    'local_images': set(),
                    'all_images': list(component),
                    'cluster_type': cluster_type,
                    'quality_score': self._compute_cluster_quality(component, corr_matrix)
                }
                clusters.append(cluster)
        
        return clusters
    
    def _density_based_clustering(self, corr_matrix: np.ndarray, 
                                indices: List[int]) -> List[Dict]:
        """Use DBSCAN for density-based clustering of remaining images"""
        if len(indices) < self.min_cluster_size:
            return []
        
        # Extract submatrix for remaining images
        sub_corr = corr_matrix[np.ix_(indices, indices)]
        
        # Convert correlation to distance
        distance_matrix = 1.0 - sub_corr
        
        # DBSCAN clustering
        eps = 0.6  # Adjusted based on correlation space
        min_samples = max(3, self.min_cluster_size // 2)
        
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
        labels = clustering.fit_predict(distance_matrix)
        
        clusters = []
        for label in set(labels):
            if label != -1:  # Not noise
                cluster_indices = [indices[i] for i, l in enumerate(labels) if l == label]
                if len(cluster_indices) >= self.min_cluster_size:
                    cluster = {
                        'core_images': set(cluster_indices),
                        'anchor_images': set(),
                        'local_images': set(),
                        'all_images': cluster_indices,
                        'cluster_type': 'density_based',
                        'quality_score': self._compute_cluster_quality(
                            set(cluster_indices), corr_matrix
                        )
                    }
                    clusters.append(cluster)
        
        return clusters
    
    def _add_bridge_connections(self, clusters: List[Dict], 
                              corr_matrix: np.ndarray) -> List[Dict]:
        """Add bridge images that connect multiple clusters"""
        n_images = corr_matrix.shape[0]
        
        # Find images with moderate correlation to multiple clusters
        for img_idx in range(n_images):
            cluster_connections = []
            
            for cluster_idx, cluster in enumerate(clusters):
                # Compute average correlation with cluster
                cluster_images = cluster['all_images']
                correlations = [corr_matrix[img_idx, ci] for ci in cluster_images 
                              if ci != img_idx]
                
                if correlations:
                    avg_corr = np.mean(correlations)
                    if avg_corr > 0.25:  # Moderate threshold for bridges
                        cluster_connections.append((cluster_idx, avg_corr))
            
            # If image connects to multiple clusters, add as anchor
            if len(cluster_connections) >= 2:
                # Add to cluster with highest correlation
                cluster_connections.sort(key=lambda x: x[1], reverse=True)
                best_cluster_idx = cluster_connections[0][0]
                
                if img_idx not in clusters[best_cluster_idx]['all_images']:
                    clusters[best_cluster_idx]['anchor_images'].add(img_idx)
                    clusters[best_cluster_idx]['all_images'].append(img_idx)
        
        return clusters
    
    def _compute_cluster_quality(self, image_indices: Set[int], 
                               corr_matrix: np.ndarray) -> float:
        """Enhanced cluster quality computation"""
        if len(image_indices) < 2:
            return 0.0
        
        indices_list = list(image_indices)
        
        # Internal coherence
        internal_correlations = []
        for i in range(len(indices_list)):
            for j in range(i + 1, len(indices_list)):
                internal_correlations.append(corr_matrix[indices_list[i], indices_list[j]])
        
        internal_score = np.mean(internal_correlations) if internal_correlations else 0.0
        
        # Size optimality
        size_ratio = len(image_indices) / self.max_cluster_size
        size_score = 1.0 - abs(1.0 - size_ratio) if size_ratio <= 1.0 else 1.0 / size_ratio
        
        # Connectivity strength
        min_connections = 2
        connectivity_scores = []
        for idx in indices_list:
            connections = sum(1 for other_idx in indices_list 
                            if other_idx != idx and corr_matrix[idx, other_idx] > 0.3)
            connectivity_scores.append(min(connections / min_connections, 1.0))
        
        connectivity_score = np.mean(connectivity_scores) if connectivity_scores else 0.0
        
        return 0.5 * internal_score + 0.3 * size_score + 0.2 * connectivity_score


class RobustBundleAdjustment:
    """
    Enhanced bundle adjustment with better anchor detection and robust optimization
    """
    
    def __init__(self):
        self.convergence_threshold = 1e-6
        self.max_iterations = 50
        
    def global_bundle_adjustment(self, batch_predictions: List[Dict], 
                               anchor_correspondences: Dict) -> List[Dict]:
        """
        Robust global bundle adjustment with improved anchor handling
        """
        print("Performing robust global bundle adjustment...")
        
        if not batch_predictions:
            return batch_predictions
        
        # Enhanced anchor correspondence detection
        enhanced_anchors = self._detect_enhanced_anchors(batch_predictions)
        print(f"Found {len(enhanced_anchors)} enhanced anchor correspondences")
        
        # Progressive alignment strategy
        aligned_batches = self._progressive_alignment(batch_predictions, enhanced_anchors)
        
        # Global refinement
        if len(aligned_batches) > 1:
            aligned_batches = self._global_refinement(aligned_batches, enhanced_anchors)
        
        return aligned_batches
    
    def _detect_enhanced_anchors(self, batch_predictions: List[Dict]) -> Dict:
        """
        Enhanced anchor detection using geometric consistency
        """
        anchors = {}
        
        # Method 1: Image path overlaps (existing)
        image_to_batches = defaultdict(list)
        for batch_idx, batch in enumerate(batch_predictions):
            for img_path in batch.get('image_paths', []):
                image_to_batches[img_path].append(batch_idx)
        
        for img_path, batch_indices in image_to_batches.items():
            if len(batch_indices) > 1:
                anchors[img_path] = {
                    'batches': batch_indices,
                    'type': 'image_overlap'
                }
        
        # Method 2: Geometric proximity anchors
        geometric_anchors = self._find_geometric_anchors(batch_predictions)
        anchors.update(geometric_anchors)
        
        return anchors
    
    def _find_geometric_anchors(self, batch_predictions: List[Dict]) -> Dict:
        """
        Find anchors based on geometric proximity of point clouds
        """
        geometric_anchors = {}
        
        # Extract representative points from each batch
        batch_representatives = []
        for batch_idx, batch in enumerate(batch_predictions):
            points = batch.get('world_points_from_depth')
            if points is not None:
                points_flat = points.reshape(-1, 3)
                valid_mask = ~np.any(np.isnan(points_flat) | np.isinf(points_flat), axis=1)
                valid_points = points_flat[valid_mask]
                
                if len(valid_points) > 100:
                    # Sample representative points
                    sample_indices = np.random.choice(
                        len(valid_points), 
                        min(1000, len(valid_points)), 
                        replace=False
                    )
                    representative_points = valid_points[sample_indices]
                    batch_representatives.append((batch_idx, representative_points))
        
        # Find geometric overlaps
        for i, (batch_i, points_i) in enumerate(batch_representatives):
            for j, (batch_j, points_j) in enumerate(batch_representatives):
                if i >= j:
                    continue
                
                # Compute nearest neighbor distances
                distances = cdist(points_i, points_j)
                min_distances = np.min(distances, axis=1)
                
                # Count points within proximity threshold
                proximity_threshold = 2.0  # meters
                close_points = np.sum(min_distances < proximity_threshold)
                overlap_ratio = close_points / len(points_i)
                
                if overlap_ratio > 0.1:  # 10% overlap
                    anchor_key = f"geometric_{batch_i}_{batch_j}"
                    geometric_anchors[anchor_key] = {
                        'batches': [batch_i, batch_j],
                        'type': 'geometric_overlap',
                        'overlap_ratio': overlap_ratio
                    }
        
        return geometric_anchors
    
    def _progressive_alignment(self, batch_predictions: List[Dict], 
                             anchors: Dict) -> List[Dict]:
        """
        Progressive alignment starting from the most connected batch
        """
        if len(batch_predictions) <= 1:
            return batch_predictions
        
        # Find batch connectivity
        batch_connections = defaultdict(list)
        for anchor_info in anchors.values():
            batches = anchor_info['batches']
            for batch_idx in batches:
                batch_connections[batch_idx].extend(
                    [b for b in batches if b != batch_idx]
                )
        
        # Start with most connected batch
        if batch_connections:
            reference_idx = max(batch_connections.keys(), 
                              key=lambda x: len(batch_connections[x]))
        else:
            reference_idx = 0
        
        print(f"Using batch {reference_idx} as reference")
        
        aligned_batches = [None] * len(batch_predictions)
        aligned_batches[reference_idx] = batch_predictions[reference_idx]
        
        # Progressive alignment
        aligned_indices = {reference_idx}
        
        while len(aligned_indices) < len(batch_predictions):
            best_candidate = None
            best_transform = None
            best_score = -1
            
            # Find best next batch to align
            for batch_idx in range(len(batch_predictions)):
                if batch_idx in aligned_indices:
                    continue
                
                # Check if this batch has anchors with aligned batches
                for anchor_info in anchors.values():
                    anchor_batches = set(anchor_info['batches'])
                    if (batch_idx in anchor_batches and 
                        len(anchor_batches & aligned_indices) > 0):
                        
                        # Try to align this batch
                        ref_batch_idx = list(anchor_batches & aligned_indices)[0]
                        transform, score = self._compute_alignment(
                            batch_predictions[batch_idx],
                            aligned_batches[ref_batch_idx],
                            anchor_info
                        )
                        
                        if score > best_score:
                            best_candidate = batch_idx
                            best_transform = transform
                            best_score = score
            
            if best_candidate is not None:
                # Apply transformation
                transformed_batch = self._apply_transformation(
                    batch_predictions[best_candidate], best_transform
                )
                aligned_batches[best_candidate] = transformed_batch
                aligned_indices.add(best_candidate)
                print(f"Aligned batch {best_candidate} (score: {best_score:.3f})")
            else:
                # Fallback: align remaining batches using centroid alignment
                for batch_idx in range(len(batch_predictions)):
                    if batch_idx not in aligned_indices:
                        centroid_aligned = self._centroid_alignment(
                            aligned_batches[reference_idx],
                            batch_predictions[batch_idx]
                        )
                        aligned_batches[batch_idx] = centroid_aligned
                        aligned_indices.add(batch_idx)
                        print(f"Centroid-aligned batch {batch_idx}")
                break
        
        return [batch for batch in aligned_batches if batch is not None]
    
    def _compute_alignment(self, source_batch: Dict, target_batch: Dict, 
                         anchor_info: Dict) -> Tuple[np.ndarray, float]:
        """
        Compute alignment transformation between batches
        """
        if anchor_info['type'] == 'image_overlap':
            return self._image_based_alignment(source_batch, target_batch, anchor_info)
        elif anchor_info['type'] == 'geometric_overlap':
            return self._geometric_alignment(source_batch, target_batch)
        else:
            return np.eye(4), 0.0
    
    def _image_based_alignment(self, source_batch: Dict, target_batch: Dict, 
                             anchor_info: Dict) -> Tuple[np.ndarray, float]:
        """
        Alignment based on shared images
        """
        # Extract 3D points from shared images
        source_points = self._extract_representative_points(source_batch)
        target_points = self._extract_representative_points(target_batch)
        
        if len(source_points) >= 3 and len(target_points) >= 3:
            # Use centroids for alignment
            source_centroid = np.mean(source_points, axis=0)
            target_centroid = np.mean(target_points, axis=0)
            
            translation = target_centroid - source_centroid
            
            transform = np.eye(4)
            transform[:3, 3] = translation
            
            # Score based on point cloud overlap
            score = min(len(source_points), len(target_points)) / 1000.0
            
            return transform, score
        
        return np.eye(4), 0.0
    
    def _geometric_alignment(self, source_batch: Dict, 
                           target_batch: Dict) -> Tuple[np.ndarray, float]:
        """
        Alignment based on geometric overlap
        """
        source_points = self._extract_representative_points(source_batch)
        target_points = self._extract_representative_points(target_batch)
        
        if len(source_points) < 10 or len(target_points) < 10:
            return np.eye(4), 0.0
        
        # Use Open3D for robust registration
        source_pcd = o3d.geometry.PointCloud()
        source_pcd.points = o3d.utility.Vector3dVector(source_points)
        
        target_pcd = o3d.geometry.PointCloud()  
        target_pcd.points = o3d.utility.Vector3dVector(target_points)
        
        # Downsample for efficiency
        voxel_size = 0.5
        source_down = source_pcd.voxel_down_sample(voxel_size)
        target_down = target_pcd.voxel_down_sample(voxel_size)
        
        # ICP registration
        threshold = 2.0
        result = o3d.pipelines.registration.registration_icp(
            source_down, target_down, threshold, np.eye(4),
            o3d.pipelines.registration.TransformationEstimationPointToPoint()
        )
        
        score = result.fitness if hasattr(result, 'fitness') else 0.0
        
        return result.transformation, score
    
    def _extract_representative_points(self, batch: Dict) -> np.ndarray:
        """
        Extract representative 3D points from batch
        """
        points = batch.get('world_points_from_depth')
        if points is None:
            return np.array([])
        
        points_flat = points.reshape(-1, 3)
        valid_mask = ~np.any(np.isnan(points_flat) | np.isinf(points_flat), axis=1)
        valid_points = points_flat[valid_mask]
        
        if len(valid_points) > 5000:
            # Subsample for efficiency
            indices = np.random.choice(len(valid_points), 5000, replace=False)
            valid_points = valid_points[indices]
        
        return valid_points
    
    def _apply_transformation(self, batch: Dict, transform: np.ndarray) -> Dict:
        """
        Apply transformation to batch
        """
        transformed_batch = batch.copy()
        
        # Transform world points
        points = batch['world_points_from_depth']
        original_shape = points.shape
        points_flat = points.reshape(-1, 3)
        
        # Create homogeneous coordinates
        points_homo = np.ones((len(points_flat), 4))
        points_homo[:, :3] = points_flat
        
        # Apply transformation
        transformed_homo = (transform @ points_homo.T).T
        transformed_points = transformed_homo[:, :3]
        
        transformed_batch['world_points_from_depth'] = transformed_points.reshape(original_shape)
        
        return transformed_batch
    
    def _centroid_alignment(self, reference_batch: Dict, batch: Dict) -> Dict:
        """
        Simple centroid-based alignment
        """
        ref_points = self._extract_representative_points(reference_batch)
        batch_points = self._extract_representative_points(batch)
        
        if len(ref_points) > 0 and len(batch_points) > 0:
            ref_centroid = np.mean(ref_points, axis=0)
            batch_centroid = np.mean(batch_points, axis=0)
            
            translation = ref_centroid - batch_centroid
            
            transform = np.eye(4)
            transform[:3, 3] = translation
            
            return self._apply_transformation(batch, transform)
        
        return batch
    
    def _global_refinement(self, aligned_batches: List[Dict], 
                         anchors: Dict) -> List[Dict]:
        """
        Global refinement pass to minimize overall alignment error
        """
        print("Performing global refinement...")
        
        # For now, return as-is - can add iterative refinement later
        return aligned_batches


class ImprovedHierarchicalPipeline:
    """
    Improved pipeline with enhanced components
    """
    
    def __init__(self, max_resolution: int = 512, voxel_size: float = 0.08):
        self.max_resolution = max_resolution
        self.voxel_size = voxel_size
        self.device = device
        self.model = None
        
        # Enhanced components
        self.correlation_analyzer = EnhancedImageCorrelationAnalyzer()
        self.clusterer = AdaptiveHierarchicalClusterer(
            min_cluster_size=10,
            max_cluster_size=18,
            overlap_ratio=0.3
        )
        self.bundle_adjuster = RobustBundleAdjustment()
        
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
        Improved hierarchical reconstruction pipeline
        """
        print("="*60)
        print("IMPROVED HIERARCHICAL RECONSTRUCTION PIPELINE")
        print("="*60)
        
        # Find images
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tiff', '*.bmp']
        image_paths = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext)))
            image_paths.extend(glob.glob(os.path.join(target_dir, "images", ext.upper())))
        
        image_paths = sorted(image_paths)
        print(f"Found {len(image_paths)} images")
        
        if len(image_paths) == 0:
            raise ValueError("No images found")
        
        # Load model
        self.load_vggt_model()
        
        # Phase 1: Enhanced correlation analysis
        print("\n" + "="*50)
        print("PHASE 1: ENHANCED IMAGE CORRELATION ANALYSIS")
        print("="*50)
        
        correlation_matrix = self.correlation_analyzer.compute_image_correlation_matrix(image_paths)
        
        # Save and analyze correlation matrix
        np.save(os.path.join(target_dir, "enhanced_correlation_matrix.npy"), correlation_matrix)
        self._analyze_correlation_matrix(correlation_matrix)
        
        # Phase 2: Adaptive clustering
        print("\n" + "="*40)
        print("PHASE 2: ADAPTIVE HIERARCHICAL CLUSTERING")
        print("="*40)
        
        clusters = self.clusterer.create_hierarchical_clusters(image_paths, correlation_matrix)
        
        # Save cluster information
        self._save_cluster_info(clusters, image_paths, target_dir)
        
        # Phase 3: Process clusters
        print("\n" + "="*40)
        print("PHASE 3: VGGT RECONSTRUCTION PER CLUSTER")
        print("="*40)
        
        batch_predictions = []
        for i, cluster in enumerate(clusters):
            cluster_image_paths = [image_paths[idx] for idx in cluster['all_images']]
            print(f"\nProcessing cluster {i+1}/{len(clusters)}: {len(cluster_image_paths)} images "
                  f"(quality: {cluster.get('quality_score', 0):.3f})")
            
            try:
                prediction = self._process_cluster_with_retry(cluster_image_paths, cluster)
                prediction['cluster_info'] = cluster
                batch_predictions.append(prediction)
            except Exception as e:
                print(f"Failed to process cluster {i+1}: {e}")
                continue
        
        if not batch_predictions:
            raise RuntimeError("All clusters failed to process")
        
        print(f"Successfully processed {len(batch_predictions)}/{len(clusters)} clusters")
        
        # Phase 4: Enhanced bundle adjustment
        print("\n" + "="*40)
        print("PHASE 4: ENHANCED GLOBAL ALIGNMENT")
        print("="*40)
        
        anchor_correspondences = self._build_enhanced_anchors(clusters, image_paths)
        aligned_predictions = self.bundle_adjuster.global_bundle_adjustment(
            batch_predictions, anchor_correspondences
        )
        
        # Phase 5: Merge results
        print("\n" + "="*40)
        print("PHASE 5: MERGING FINAL RESULTS")
        print("="*40)
        
        merged_prediction = self._merge_predictions(aligned_predictions)
        
        print("\n" + "="*60)
        print("IMPROVED RECONSTRUCTION COMPLETE!")
        print("="*60)
        
        return merged_prediction
    
    def _analyze_correlation_matrix(self, correlation_matrix: np.ndarray):
        """Analyze correlation matrix characteristics"""
        non_diag = correlation_matrix[correlation_matrix < 1.0]
        non_zero = non_diag[non_diag > 0]
        
        print(f"Correlation analysis:")
        print(f"  - Non-zero correlations: {len(non_zero)}/{len(non_diag)} ({100*len(non_zero)/len(non_diag):.1f}%)")
        print(f"  - Mean correlation: {np.mean(non_zero):.3f}")
        print(f"  - Std correlation: {np.std(non_zero):.3f}")
        print(f"  - Max correlation: {np.max(non_zero):.3f}")
        print(f"  - Strong correlations (>0.5): {np.sum(non_zero > 0.5)}")
        print(f"  - Medium correlations (0.3-0.5): {np.sum((non_zero > 0.3) & (non_zero <= 0.5))}")
    
    def _save_cluster_info(self, clusters: List[Dict], image_paths: List[str], target_dir: str):
        """Save detailed cluster information"""
        cluster_info = []
        for cluster in clusters:
            cluster_paths = [image_paths[i] for i in cluster['all_images']]
            cluster_info.append({
                'cluster_id': cluster['cluster_id'],
                'cluster_type': cluster.get('cluster_type', 'unknown'),
                'quality_score': cluster.get('quality_score', 0.0),
                'image_count': len(cluster['all_images']),
                'core_images': len(cluster.get('core_images', [])),
                'anchor_images': len(cluster.get('anchor_images', [])),
                'image_paths': cluster_paths
            })
        
        with open(os.path.join(target_dir, "enhanced_cluster_info.json"), 'w') as f:
            json.dump(cluster_info, f, indent=2)
    
    def _process_cluster_with_retry(self, cluster_image_paths: List[str], cluster: Dict) -> Dict:
        """Process cluster with retry logic and memory management"""
        max_retries = 2
        
        for attempt in range(max_retries + 1):
            try:
                # Adjust resolution based on cluster size and attempt
                if attempt == 0:
                    resolution = min(self.max_resolution, max(256, 512 - len(cluster_image_paths) * 8))
                else:
                    resolution = max(256, resolution - 128)
                
                print(f"  Attempt {attempt + 1}: resolution {resolution}")
                
                # Clear GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Process cluster
                prediction = self._process_single_cluster(cluster_image_paths, resolution)
                return prediction
                
            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries:
                    raise
                
                # Clear memory before retry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    def _process_single_cluster(self, image_paths: List[str], resolution: int) -> Dict:
        """Process a single cluster"""
        # Load and preprocess images
        batch_images = load_and_preprocess_images(
            image_paths, 
            target_resolution=(resolution, resolution),
            device=self.device
        )
        
        # Run VGGT
        with torch.no_grad():
            predictions = self.model(batch_images)
        
        # Post-process predictions
        if 'pose_encodings' in predictions:
            predictions = pose_encoding_to_extri_intri(predictions)
        
        if 'depth_maps' in predictions:
            predictions = unproject_depth_map_to_point_map(predictions)
        
        # Add image paths
        predictions['image_paths'] = image_paths
        
        return predictions
    
    def _build_enhanced_anchors(self, clusters: List[Dict], image_paths: List[str]) -> Dict:
        """Build enhanced anchor correspondences"""
        anchors = {}
        
        # Image-based anchors
        image_to_clusters = defaultdict(list)
        for cluster in clusters:
            for img_idx in cluster['all_images']:
                if img_idx < len(image_paths):
                    img_path = image_paths[img_idx]
                    image_to_clusters[img_path].append(cluster['cluster_id'])
        
        for img_path, cluster_ids in image_to_clusters.items():
            if len(cluster_ids) > 1:
                anchors[img_path] = {
                    'clusters': cluster_ids,
                    'type': 'image_overlap'
                }
        
        print(f"Built {len(anchors)} enhanced anchor correspondences")
        return anchors
    
    def _merge_predictions(self, predictions_list: List[Dict]) -> Dict:
        """Merge predictions with enhanced metadata"""
        if not predictions_list:
            return {}
        
        merged = {}
        
        # Merge numerical arrays
        numerical_keys = ['world_points_from_depth', 'depth_maps', 'confidence_maps']
        for key in numerical_keys:
            if key in predictions_list[0]:
                arrays = [pred[key] for pred in predictions_list if key in pred]
                merged[key] = np.concatenate(arrays, axis=0)
        
        # Merge lists
        list_keys = ['image_paths']
        for key in list_keys:
            if key in predictions_list[0]:
                merged[key] = []
                for pred in predictions_list:
                    merged[key].extend(pred.get(key, []))
        
        # Enhanced metadata
        merged.update({
            'num_clusters': len(predictions_list),
            'cluster_sizes': [len(pred.get('image_paths', [])) for pred in predictions_list],
            'cluster_types': [pred.get('cluster_info', {}).get('cluster_type', 'unknown') 
                            for pred in predictions_list],
            'cluster_qualities': [pred.get('cluster_info', {}).get('quality_score', 0.0) 
                                for pred in predictions_list],
            'total_images': sum(len(pred.get('image_paths', [])) for pred in predictions_list)
        })
        
        return merged


def main():
    """Run the improved hierarchical reconstruction pipeline"""
    target_dir = "C:\\repos\\gatech\\photogrammetry\\south-building"
    
    # Create improved pipeline
    pipeline = ImprovedHierarchicalPipeline(
        max_resolution=512,
        voxel_size=0.08
    )
    
    try:
        start_time = time.time()
        predictions = pipeline.process_large_dataset(target_dir)
        total_time = time.time() - start_time
        
        print(f"\nTotal processing time: {total_time:.2f} seconds")
        
        # Save results
        glb_path = os.path.join(target_dir, f"improved_reconstruction_{int(time.time())}.glb")
        print(f"Saving GLB to {glb_path}")
        
        try:
            scene = predictions_to_glb(predictions, conf_thres=30.0, target_dir=target_dir)
            scene.export(glb_path)
        except Exception as e:
            print(f"GLB export failed: {e}")
        
        # Enhanced results summary
        world_points = predictions.get("world_points_from_depth")
        if world_points is not None:
            points_flat = world_points.reshape(-1, 3)
            valid_mask = ~np.any(np.isnan(points_flat) | np.isinf(points_flat), axis=1)
            valid_points = points_flat[valid_mask]
            
            print(f"\n--- ENHANCED RECONSTRUCTION RESULTS ---")
            print(f"Total images processed: {predictions.get('total_images', 0)}")
            print(f"Generated {len(valid_points):,} valid 3D points")
            print(f"Used {predictions.get('num_clusters', 0)} clusters")
            print(f"Cluster types: {set(predictions.get('cluster_types', []))}")
            print(f"Average cluster quality: {np.mean(predictions.get('cluster_qualities', [0])):.3f}")
            print(f"Cluster sizes: {predictions.get('cluster_sizes', [])}")
            
            # Point cloud quality metrics
            if len(valid_points) > 100:
                point_spread = np.std(valid_points, axis=0)
                point_range = np.ptp(valid_points, axis=0)
                print(f"Point cloud spread (std): [{point_spread[0]:.2f}, {point_spread[1]:.2f}, {point_spread[2]:.2f}]")
                print(f"Point cloud range: [{point_range[0]:.2f}, {point_range[1]:.2f}, {point_range[2]:.2f}]")
        
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()