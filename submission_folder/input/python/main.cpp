#include <iostream>
#include <fstream>
#include <array>
#include <algorithm>
#include <vector>
#include <random>
#include <sstream>
#include <cassert>

using namespace std;
typedef float feature_type;
typedef vector<vector<int>> grid;
struct box {
    int xmin = 999, ymin = 999, xmax = -999, ymax = -999;
    double width()  const {return xmin != 999 && xmax != -999 ? xmax - xmin : 0.0;}
    double height() const {return ymin != 999 && ymax != -999 ? ymax - ymin : 0.0;}
    double area() const {return width()*height();}
    double perimeter() const {return 2*(width()+height());}
    static box grid(const grid& g) {return box{0, 0, int(g.size()), int(g[0].size())};}
    box reshape(int t) const {return box{xmin-t, ymin-t, xmax+t, ymax+t};}
    bool has_box(box b) const {
        return area() > 0 && b.area() > 0 && xmin <= b.xmin && xmax >= b.xmax && ymin <= b.ymin && ymax >= b.ymax;
    }
    bool has_intersection(box b) const {
        return area() > 0 && b.area() > 0 && ymin < b.ymax && ymax > b.ymin && xmin < b.xmax && xmax > b.xmin;
    }
    double iou(box b) const {
        double xmaxmin = max(xmin, b.xmin);
        double ymaxmin = max(ymin, b.ymin);
        double xminmax = min(xmax, b.xmax);
        double yminmax = min(ymax, b.ymax);
        
        bool has_inter = has_intersection(b);
        double inter_area = has_inter ? (xminmax - xmaxmin) * (yminmax - ymaxmin) : 0.0;
        double whole_area = area() + b.area() - inter_area;
        return inter_area / whole_area;
    }
};
vector<string> split(istream& ss, char sep = ' ') {
    vector<string> output;
    string line;
    for (;getline(ss, line, sep);) {
        output.emplace_back(line);
    }
    return output;
}
vector<string> split(string input, char sep = ' ') {
    istringstream ss(input);
    return split(ss, sep);
}
array<int, 10> count(const grid& g, box b) {
    array<int, 10> result;
    result.fill(0);
    for (auto x = b.xmin; x < b.xmax; ++x)
        for (auto y = b.ymin; y < b.ymax; ++y)
            ++result[g[x][y]];
    return result;
}
array<int, 10> count(const grid& g) {
    return count(g, box::grid(g));
}
bool has_vertical_symmetry(const grid& g, box b) {
    for (int x = b.xmin; x<b.xmax; ++x)
        for (int dy = 0; dy < (b.ymax-b.ymin)/2; ++dy) {
            if (g[x][b.ymin+dy] != g[x][b.ymax-dy-1])
                return false;
        }
    return true;
}
bool has_horizontal_symmetry(const grid& g, box b) {
    for (int y = b.ymin; y < b.ymax; ++y)
        for (int dx = 0; dx < (b.xmax-b.xmin)/2; ++dx) {
            if (g[b.xmin+dx][y] != g[b.xmax-dx-1][y])
                return false;
        }
    return true;
}
bool has_frame(const grid& g, box b, bool unique_frame = false) {
    vector<int> cs;
    int mx = int(g.size()), my = int(g[0].size());
    int xmin_ = max(0, b.xmin), xmax_ = min(b.xmax, mx);
    int ymin_ = max(0, b.ymin), ymax_ = min(b.ymax, my);
    if (b.xmin == xmin_)
        for (int y = ymin_; y < ymax_; ++y)
            cs.emplace_back(g[b.xmin][y]);
    if (b.xmax == xmax_)
        for (int y = ymin_; y < ymax_; ++y)
            cs.emplace_back(g[b.xmax-1][y]);
    if (b.ymin == ymin_)
        for (int x = xmin_; x < xmax_; ++x)
            cs.emplace_back(g[x][b.ymin]);
    if (b.ymax == ymax_)
        for (int x = xmin_; x < xmax_; ++x)
            cs.emplace_back(g[x][b.ymax-1]);
    for (int i = 1; i < cs.size(); ++i)
        if (cs[i] != cs[i-1])
            return false;
    if (unique_frame && !cs.empty())
        for (int x = max(0, b.xmin+1); x < min(b.xmax-1, mx); ++x)
            for (int y = max(0, b.ymin+1); y < min(b.ymax-1, my); ++y)
                if (g[x][y] == cs[0])
                    return false;
    return true;
}
int cnt_strime(const grid& g, box b) {
    int n = 0;
    int mx = int(g.size()), my = int(g[0].size());
    if (b.xmin >= b.xmax || b.ymin >= b.ymax)
        return n;
    int xmin_ = max(0, b.xmin), xmax_ = min(b.xmax, mx);
    int ymin_ = max(0, b.ymin), ymax_ = min(b.ymax, my);
    if (b.xmin == xmin_ && ymax_ - ymin_ > 1) {
        ++n;
        for (int y = ymin_+1; y < ymax_; ++y)
            if (g[b.xmin][y-1] != g[b.xmin][y]) {
                --n;
                break;
            }
    }
    if (b.xmax == xmax_ && ymax_ - ymin_ > 1) {
        ++n;
        for (int y = ymin_+1; y < ymax_; ++y)
            if (g[b.xmax-1][y-1] != g[b.xmax-1][y]) {
                --n;
                break;
            }
    }
    if (b.ymin == ymin_ && xmax_ - xmin_ > 1) {
        ++n;
        for (int x = xmin_+1; x < xmax_; ++x)
            if (g[x-1][b.ymin] != g[x][b.ymin]) {
                --n;
                break;
            }
    }
    if (b.ymax == ymax_ && xmax_ - xmin_ > 1) {
        ++n;
        for (int x = xmin_+1; x < xmax_; ++x)
            if (g[x-1][b.ymax-1] != g[x][b.ymax-1]) {
                --n;
                break;
            }
    }
    return n;
}
bool is_same_box(const grid& g, box l, box r) {
    for (int dx = 0; dx < l.width(); ++dx)
        for (int dy = 0; dy < l.height(); ++dy)
            if (g[l.xmin+dx][l.ymin+dy] != g[r.xmin+dx][r.ymin+dy])
                return false;
    return true;
}
int cnt_same_boxes(const grid& g, box b) {
    int n = 0;
    int width = b.width();
    int height = b.height();
    for (int x = 0; x < g.size() - width; ++x)
        for (int y = 0; y < g[0].size() - height; ++y) {
            if (is_same_box(g, b, {x, y, width, height}))
                ++n;
        }
    return n;
}
array<box, 10> get_boxes_of_colors(const grid& g) {
    array<box, 10> boxes;
    for (int x = 0; x < g.size(); ++x)
        for (int y = 0; y < g[0].size(); ++y) {
            int c = g[x][y];
            boxes[c].xmin = min(boxes[c].xmin, x);
            boxes[c].ymin = min(boxes[c].ymin, y);
            boxes[c].xmax = max(boxes[c].xmax, x+1);
            boxes[c].ymax = max(boxes[c].ymax, y+1);
        }
    return boxes;
}
array<box, 10> get_boxes_of_colors_inverse(const grid& g) {
    array<box, 10> boxes;
    for (int x = 0; x < g.size(); ++x)
        for (int y = 0; y < g[0].size(); ++y) {
            for (int c = 0; c < 10; ++c) if (c != g[x][y]) {
                boxes[c].xmin = min(boxes[c].xmin, x);
                boxes[c].ymin = min(boxes[c].ymin, y);
                boxes[c].xmax = max(boxes[c].xmax, x+1);
                boxes[c].ymax = max(boxes[c].ymax, y+1);
            }
        }
    return boxes;
}
void boxes_features(vector<feature_type>& row, box l, box r) {
//    row.emplace_back(l.area()/r.area());
//    row.emplace_back(l.iou(r));
    row.emplace_back(l.iou(r) > 0.99);
}
vector<int> get_colors(const grid& g, const array<box, 10>& boxes_of_colors, box bx) {
    vector<int> colors;
    auto cnt_colors = count(g, bx);
    auto all_colors = count(g);
    int used_color = -1;
    int used_color2 = -1;
    for (int  c = 9; c >= 0; --c) {
        if (used_color != -1 && cnt_colors[c] > 0) {
            used_color2 = c;
            break;
        }
        if (used_color == -1 && cnt_colors[c] > 0) {
            used_color = c;
        }
    }
    int gr_percent = used_color;
    int gr_area_not_black = used_color;
    int gr_area = used_color;
    int ls_area = used_color;
    int gr_iou = used_color;
    for (int c = 0; c < 10; ++c) {
//        colors.emplace_back(c);
        if (cnt_colors[gr_percent] / float(all_colors[gr_percent]) < cnt_colors[c] / float(all_colors[c]))
            gr_percent = c;
        if (boxes_of_colors[gr_area].area() < boxes_of_colors[c].area())
            gr_area = c;
        if (c != 0 && boxes_of_colors[gr_area_not_black].area() < boxes_of_colors[c].area())
            gr_area_not_black = c;
        if (boxes_of_colors[c].area() > 0 && boxes_of_colors[ls_area].area() > boxes_of_colors[c].area())
            ls_area = c;
        if (boxes_of_colors[gr_iou].iou(bx) < boxes_of_colors[c].iou(bx))
            gr_iou = c;
    }
    int gr_area2 = gr_area == used_color ? used_color2 : used_color;
    for (int c = 0; c < 10; ++c) {
        if (c != gr_area && boxes_of_colors[gr_area2].area() < boxes_of_colors[c].area())
            gr_area2 = c;
    }
    colors.emplace_back(gr_percent);        // 0
    colors.emplace_back(gr_area_not_black); // 1
    colors.emplace_back(gr_area);           // 2
    colors.emplace_back(gr_area2);          // 3
    colors.emplace_back(ls_area);           // 4
    colors.emplace_back(gr_iou);            // 5
    
    return colors;
}
vector<feature_type> make_feature(const grid& g, const array<box, 10>& boxes_of_colors, const box bx) {
    vector<feature_type> row;
    row.emplace_back(bx.xmin);
    row.emplace_back(bx.ymin);
    row.emplace_back(bx.xmax);
    row.emplace_back(bx.ymax);
    
    auto ibx = box::grid(g);
    
    int has_boxes = 0;
    int in_boxes = 0;
    auto boxes_of_colors_inverse = get_boxes_of_colors_inverse(g);
    for (auto c : get_colors(g, boxes_of_colors, bx)) {
        boxes_features(row, bx, boxes_of_colors[c]);
        boxes_features(row, bx, boxes_of_colors_inverse[c]);
        boxes_features(row, bx.reshape(1), boxes_of_colors[c]);
        boxes_features(row, bx.reshape(1), boxes_of_colors_inverse[c]);
    }
    auto cnt_colors = count(g, bx);
    int ucnt_colors = 0;
    for (int c = 0; c < 10; ++c) {
        ucnt_colors += cnt_colors[c] > 0;
        has_boxes += bx.has_box(boxes_of_colors[c]);
        in_boxes += boxes_of_colors[c].has_box(bx);
    }
    
    boxes_features(row, bx, ibx);
    bool has_frame_ = has_frame(g, bx);
    bool has_frame_1 = has_frame(g, bx.reshape(1));
//    bool has_frame_m1 = has_frame(g, bx.reshape(-1));
    int cnt_trime_ = cnt_strime(g, bx);
    row.emplace_back(cnt_same_boxes(g, bx));
    row.emplace_back(has_frame_ ? cnt_same_boxes(g, bx) : 0);
    row.emplace_back(cnt_trime_ == 0 ? cnt_same_boxes(g, bx) : 0);
    row.emplace_back(has_vertical_symmetry(g, bx));
    row.emplace_back(has_horizontal_symmetry(g, bx));

    row.emplace_back(ucnt_colors);
    row.emplace_back(has_boxes);
    row.emplace_back(in_boxes);
    row.emplace_back(has_frame(g, bx, true));
    row.emplace_back(has_frame(g, bx.reshape(1), true));
    row.emplace_back(has_frame_);
    row.emplace_back(has_frame_1);
//    row.emplace_back(has_frame_m1);
    row.emplace_back(has_frame_1 || has_frame_);
    row.emplace_back(has_frame_1 && has_frame_);
    row.emplace_back(has_frame_1 == has_frame_);
    row.emplace_back(bx.width());
    row.emplace_back(bx.height());
    row.emplace_back(bx.area());
    row.emplace_back(cnt_trime_);
    row.emplace_back(cnt_strime(g, bx.reshape(1)));
    row.emplace_back(cnt_strime(g, bx.reshape(-1)));
    
//    row.emplace_back(perimeter);
    return row;
}
string get_columns() {
    stringstream ss;
    ss << "xmin" << "\t";
    ss << "ymin" << "\t";
    ss << "xmax" << "\t";
    ss << "ymax" << "\t";
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 1 + 3*(i < 6); ++j) {
//            ss << "[" << i << j << "] div_areas" << "\t";
//            ss << "[" << i << j << "] iou" << "\t";
            ss << "[" << i << j << "] iou_1" << "\t";
        }
    }
    ss << "cnt_same_boxes" << "\t";
    ss << "cnt_same_boxes_w_fr" << "\t";
    ss << "cnt_same_boxes_wo_tr" << "\t";
    ss << "has_vertical_symmetry" << "\t";
    ss << "has_horizontal_symmetry" << "\t";
    
    ss << "ucnt_colors" << "\t";
    
    ss << "has_boxes" << "\t";
    ss << "in_boxes" << "\t";
    ss << "has_uframe" << "\t";
    ss << "has_uframe_1" << "\t";
    ss << "has_frame" << "\t";
    ss << "has_frame_1" << "\t";
//    ss << "has_frame_1m" << "\t";
    ss << "has_frame_or" << "\t";
    ss << "has_frame_and" << "\t";
    ss << "has_frame_eq" << "\t";
    ss << "width" << "\t";
    ss << "height" << "\t";
    ss << "area" << "\t";
    ss << "cnt_strim" << "\t";
    ss << "cnt_strim_1" << "\t";
    ss << "cnt_strim_m1";
//    ss << "perimeter";
    return ss.str();
}
void make_features(const grid& g, ostream& out) {
    auto boxes_of_colors = get_boxes_of_colors(g);
    int n = 0;
    box l = box::grid(g);
    for (int xmin = 0; xmin < g.size(); ++xmin)
        for (int ymin = 0; ymin < g[0].size(); ++ymin)
            for (int xmax = xmin+1; xmax < g.size()+1; ++xmax)
                for (int ymax = ymin+1; ymax < g[0].size()+1; ++ymax) {
                    box r = {xmin, ymin, xmax, ymax};
                    if (r.area() == l.area()) // || r.area() == 1) || (!has_frame(g, r) && !has_frame(g, r.reshape(1)))
                        continue;
                    auto row = make_feature(g, boxes_of_colors, r);
                    out.write((char*)&row[0], row.size() * sizeof(row[0]));
                    n += 1;
                }
    cout << "rows: " << n << endl;
}
inline bool exists(const std::string& name) {
    ifstream f(name.c_str());
    return f.good();
}
int main() {
    string dir = "jupyter/arc/";
    if (!exists(dir+"ex.txt"))
        dir = "./";
    vector<grid> inputs;
    ifstream fin(dir + "ex.txt");
    ofstream fout(dir + "features.bin", ios::out | ios::binary);
    ofstream fcolumns(dir + "features.tsv");
    fcolumns << get_columns();
    for (auto input: split(fin, ' ')) {
        vector<vector<int>> g;
        for (auto line : split(input, '|')) {
            vector<int> row;
            for (char& c : line)
                row.emplace_back(c-'0');
            g.emplace_back(row);
        }
        inputs.emplace_back(g);
    }
    cout << "inputs: " << inputs.size() << endl;
    auto features = make_feature({{1}}, get_boxes_of_colors({{1}}),{0, 0, 1, 1});
    cout << "features: " << features.size() << endl;
    cout << "columns: " << split(get_columns(), '\t').size() << endl;
    assert(features.size() == split(get_columns(), '\t').size());
    for (auto input : inputs) {
        cout << "shape: " << input.size() << "x" << input[0].size() << endl;
        make_features(input, fout);
    }
    return 0;
}
