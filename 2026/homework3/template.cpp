#include <cstddef>
#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <limits>
#include <iomanip>

// Include ParlayLib (adjust the path if needed)
#include "parlaylib/include/parlay/primitives.h"
#include "parlaylib/include/parlay/parallel.h"
#include "parlaylib/include/parlay/range.h"
#include "parlaylib/include/parlay/sequence.h"
#include "parlaylib/include/parlay/utilities.h"

// A simple 2D point structure
struct Point2D {
  double x, y;

  // constructor
  Point2D(double xx=0.0, double yy=0.0) : x(xx), y(yy) {}
};

// A helper to compute squared distance
inline double squared_distance(const Point2D& a, const Point2D& b) {
  double dx = a.x - b.x;
  double dy = a.y - b.y;
  return dx*dx + dy*dy;
}

// KD-Tree node
struct KDNode {
  int axis;          // 0 for x, 1 for y
  double splitValue; // coordinate pivot
  int pointIndex;    // index in the original array

  KDNode* left;      // pointer to left child
  KDNode* right;     // pointer to right child

  // constructor
  KDNode() : axis(0), splitValue(0.0), pointIndex(-1),
             left(nullptr), right(nullptr) {}
};

// DistIndex for storing (distance^2, index)
struct DistIndex {
  double dist;
  int index;
  DistIndex(double d=0, int i=0) : dist(d), index(i) {}
};

// For a max-heap, we want to put the largest distance on top
inline bool operator<(const DistIndex &a, const DistIndex &b) {
  return a.dist < b.dist;
}

KDNode* build_kd_tree(
    parlay::slice<int*, int*> indices,
    const parlay::sequence<Point2D>& points,
    int depth = 0
) {
  size_t n = indices.size();
  if (n == 0) {
    return nullptr;
  }
  if (n == 1) {
    int idx = indices[0];
    KDNode* leaf = new KDNode{};
    leaf->axis = depth % 2;
    leaf->splitValue = (leaf->axis == 0) ? points[idx].x : points[idx].y;
    leaf->pointIndex = idx;
    return leaf;
  }

  int axis = depth % 2;
  parlay::sort_inplace(indices, [&](int a, int b) {
    double va = (axis == 0) ? points[a].x : points[a].y;
    double vb = (axis == 0) ? points[b].x : points[b].y;
    if (va != vb) return va < vb;
    return a < b;
  });

  size_t mid = n / 2;
  int med_idx = indices[mid];
  double split = (axis == 0) ? points[med_idx].x : points[med_idx].y;

  KDNode* left_child = nullptr;
  KDNode* right_child = nullptr;

  if (mid > 0 && (n - mid - 1) > 0) {
    parlay::par_do(
        [&] {
          left_child = build_kd_tree(indices.cut(0, mid), points, depth + 1);
        },
        [&] {
          right_child =
              build_kd_tree(indices.cut(mid + 1, n), points, depth + 1);
        });
  } else if (mid > 0) {
    left_child = build_kd_tree(indices.cut(0, mid), points, depth + 1);
  } else {
    right_child =
        build_kd_tree(indices.cut(mid + 1, n), points, depth + 1);
  }

  KDNode* node = new KDNode{};
  node->axis = axis;
  node->splitValue = split;
  node->pointIndex = med_idx;
  node->left = left_child;
  node->right = right_child;
  return node;
}

// KNN Helper: holds a local max-heap of size k
class KNNHelper {
public:
  KNNHelper(const parlay::sequence<Point2D>& pts, int kk)
    : points(pts), k(kk) {
    best.reserve(k);
  }

  void search(const KDNode* node, const Point2D& q) {
    if (!node) return;

    if (node->pointIndex >= 0) {
      double d2 = squared_distance(q, points[node->pointIndex]);
      update_best(d2, node->pointIndex);
    }

    if (!node->left && !node->right) return;

    double qcoord = (node->axis == 0) ? q.x : q.y;
    const KDNode* near_sub = nullptr;
    const KDNode* far_sub = nullptr;
    if (qcoord < node->splitValue) {
      near_sub = node->left;
      far_sub = node->right;
    } else {
      near_sub = node->right;
      far_sub = node->left;
    }

    if (near_sub) search(near_sub, q);

    double diff = qcoord - node->splitValue;
    double plane_d2 = diff * diff;
    double worst =
        ((int)best.size() < k)
            ? std::numeric_limits<double>::infinity()
            : best[0].dist;
    if (plane_d2 < worst && far_sub) search(far_sub, q);
  }

  // Return final results sorted by ascending distance
  parlay::sequence<DistIndex> get_results() const {
    parlay::sequence<DistIndex> result(best.begin(), best.end());
    parlay::sort_inplace(result, [&](auto &a, auto &b){
      return a.dist < b.dist;
    });
    return result;
  }

private:
  const parlay::sequence<Point2D>& points;
  int k;
  std::vector<DistIndex> best; // will be a max-heap

  void update_best(double dist2, int idx) {
    auto cmp = [](const DistIndex& a, const DistIndex& b) {
      return a.dist < b.dist;
    };
    if ((int)best.size() < k) {
      best.push_back(DistIndex(dist2, idx));
      std::push_heap(best.begin(), best.end(), cmp);
    } else if (dist2 < best[0].dist) {
      std::pop_heap(best.begin(), best.end(), cmp);
      best.back() = DistIndex(dist2, idx);
      std::push_heap(best.begin(), best.end(), cmp);
    }
  }
};

// Parallel k-NN for all queries
parlay::sequence<parlay::sequence<DistIndex>>
knn_search_all(const KDNode* root,
               const parlay::sequence<Point2D>& data_points,
               const parlay::sequence<Point2D>& query_points,
               int k) {
  int Q = (int)query_points.size();
  parlay::sequence<parlay::sequence<DistIndex>> results(Q);

  parlay::parallel_for(0, Q, [&](int i){
    KNNHelper helper(data_points, k);
    helper.search(root, query_points[i]);
    results[i] = helper.get_results();
  });

  return results;
}

parlay::sequence<Point2D> load_points_from_file(const std::string &filename) {
  std::ifstream in(filename);
  if (!in) return {};
  int n = 0;
  in >> n;
  parlay::sequence<Point2D> pts;
  pts.reserve(n);
  for (int i = 0; i < n; i++) {
    double x, y;
    in >> x >> y;
    pts.push_back(Point2D(x, y));
  }
  return pts;
}

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "Usage: " << argv[0]
              << " <data_file> <query_file> <k>\n";
    return 1;
  }

  std::string data_file  = argv[1];
  std::string query_file = argv[2];
  int k = std::stoi(argv[3]);

  auto data_points = load_points_from_file(data_file);
  int N = (int)data_points.size();

  parlay::sequence<int> indices(N);
  parlay::parallel_for(0, N, [&](int i){ indices[i] = i; });
  KDNode* root = build_kd_tree(indices.cut(0, N), data_points, 0);

  auto query_points = load_points_from_file(query_file);
  int Q = (int)query_points.size();

  auto results = knn_search_all(root, data_points, query_points, k);

  std::cout << std::fixed << std::setprecision(2);
  for (int q = 0; q < Q; q++) {
    std::cout << "Query " << q << ": ("
              << query_points[q].x << ", "
              << query_points[q].y << ")\n";
    std::cout << "  kNN: ";
    for (auto &di : results[q]) {
      std::cout << "(dist2=" << di.dist
                << ", idx=" << di.index << ") ";
    }
    std::cout << "\n";
  }

  return 0;
}

