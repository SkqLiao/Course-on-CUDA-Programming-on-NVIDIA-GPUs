#include <bits/stdc++.h>

using namespace std;

int n = 8;

void print(int x, vector<vector<int>> &v) {
  cout << "after " << x << " steps: \n";
  for (int i = 0; i < n; ++i) {
    sort(v[i].begin(), v[i].end());
    cout << i << " : ";
    for (auto x : v[i]) cout << x << " ";
    cout << endl;
  }
}

void func1() {
  vector<vector<int>> v(n);
  for (int i = 0; i < n; ++i) v[i].push_back(i);
  for (int x = 1; x < n; x *= 2) {
    vector<vector<int>> tmp(v);
    for (int i = 0; i < n; ++i) {
      v[i].insert(v[i].end(), tmp[i ^ x].begin(), tmp[i ^ x].end());
    }
    print(x, v);
  }
}

void func2() {
  vector<vector<int>> v(n);
  for (int i = 0; i < n; ++i) v[i].push_back(i);
  for (int x = n / 2; x > 0; x /= 2) {
    vector<vector<int>> tmp(v);
    for (int i = 0; i < n; ++i) {
      if (i + x < n)
        v[i].insert(v[i].end(), tmp[i + x].begin(), tmp[i + x].end());
    }
    print(x, v);
  }
}

int main() {
  // func1();
  func2();
  return 0;
}