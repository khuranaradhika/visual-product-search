# Visual Product Search for E-commerce

This repository contains the midterm project for **CS7180 - Applied Deep Learning** at Northeastern University, Spring 2026.

## Problem Statement
Users often struggle to find relevant products when searching with text alone. This is particularly true when product metadata is incomplete, inconsistent, or fails to capture visual attributes such as style, pattern, or shape. This discrepancy leads to poor search relevance and missed opportunities for product discovery.

## Project Objective
We are developing a visual similarity engine for e-commerce platforms. The goal is to allow users to discover products that are visually similar to a query image, moving beyond the limitations of text-based metadata.

### **Who We Are Solving For**
Users of e-commerce platforms who require a more intuitive way to explore catalogs based on visual characteristics rather than just keywords.

## Success Metrics
To determine the effectiveness of our model, we measure:
* **Search Relevance**: Evaluation using **Precision@K** and **Recall@K** compared to a text-only baseline.
* **User Engagement**: Analysis of simulated or actual interaction rates with returned visual results.

## Technical Approach
The system utilizes deep learning to bridge the gap between pixel data and semantic product categories.



* **Feature Extraction**: Leveraging Convolutional Neural Networks (CNNs) or Vision Transformers to extract high-dimensional embeddings from product images.
* **Similarity Search**: Implementing efficient vector search to find the nearest neighbors in the embedding space.
* **Evaluation Pipeline**: A benchmarking suite to compare visual search results against traditional keyword-based metadata
