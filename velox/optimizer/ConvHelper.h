#include <iostream>
#include <random>
#include <functional>
#include <cassert>
#include <map>
#include <memory>

class Matrix3D {
public:

  int x;
  int y;
  int z;
  std::vector<float*> matrices; // size z

  Matrix3D() {}
  Matrix3D(int x, int y, int z) : x(x), y(y), z(z) { matrices.resize(z); }

  void addChannel(float* data) {
    matrices.push_back(data);
  }

  void setMatrixAtIndex(int i, float* data) {
    assert(i < z);
    matrices[i] = data;
  }

  float* getMatrixAtIndex(int i) {
    assert(i < z);
    return matrices[i];
  }
};

class Image : public Matrix3D {
public:
  int index;

  Image() {}
  Image(int index, int x, int y, int z) : Matrix3D(x, y, z), index(index)  {}

  int &getKey() { return index; }

  Image &getValue() { return *this; }

  int get_x(int padding) { return x + padding * 2; }

  int get_y(int padding) { return y + padding * 2; }

  int get_h_slides(int padding, int kernel_size, int stride) {
    return floor((get_y(padding) - kernel_size) / stride) + 1;
  }

  int get_v_slides(int padding, int kernel_size, int stride) {
    return floor((get_x(padding) - kernel_size) / stride) + 1;
  }

  float get_win_value(int window, int i, int j, int stride, int channel,
                       int kernel_size, int padding) {
    int h_slides = get_h_slides(padding, kernel_size, stride);
    int v_slides = get_v_slides(padding, kernel_size, stride);

    // calculate window origin
    int r = floor(window / h_slides);
    int c = window % h_slides;

    // Find where i,j lies in the kernel window
    // Also adjust for padding
    // We assume that padding is out of bounds of the actual array
    int x_index = ((r * stride) + i) - padding;
    int y_index = ((c * stride) + j) - padding;

    if (padding > 0 &&
        (x_index < 0 || y_index < 0 || x_index >= x || y_index >= y))
      return 0;

    int flat_index = x_index * y + y_index;
    return matrices[channel][flat_index];
  }

  int get_conv2d_matrix_rows(int kernel_size, int stride, int padding) {
    int v_slides = get_v_slides(padding, kernel_size, stride);
    int h_slides = get_h_slides(padding, kernel_size, stride);

    return v_slides * h_slides;
  }

  int get_conv2d_matrix_cols(int kernel_size, int stride, int padding) {
    return z * kernel_size * kernel_size;
  }

  int get_conv2d_window_count(int kernel_size, int stride, int padding) {
    int h_slides = get_h_slides(padding, kernel_size, stride);
    int v_slides = get_v_slides(padding, kernel_size, stride);
    // std::cout << "[IMAGE] h_slides: " << h_slides << ", v_slides: " << v_slides << std::endl;
    // std::cout << "[IMAGE] kernel: " << kernel_size << ", strides: " << stride << ", padding: " << padding << std::endl;

    return h_slides * v_slides;
  }

  int get_num_channels() { return z; }

};

class ImageChunk {
public:
  int block_row;
  int y_index;
  std::vector<float> chunk;
  int chunk_size;
  int y_size;

  int block_row_start;

  ImageChunk() {}

  ImageChunk(int size, int y_size, int block_row, int y_index, int block_row_start)
      : block_row(block_row), y_index(y_index), chunk_size(size), y_size(y_size),  block_row_start(block_row_start) {
    chunk.reserve(size);
  }

  int getChunkActualRowIndex() { return block_row_start + block_row; }

  std::vector<float> &getChunk() { return chunk; }

  int getSize() { return chunk.size(); }
};

class ImageBlock {

public:
  int block_x_index;
  int block_x_size;
  int block_y_size;
  int block_y_index;
  std::map<int, std::vector<float>> chunks;
  long key;

  int feature_count;

  // Default constructor:
  ImageBlock() {}

  // Default destructor:
  ~ImageBlock() {}

  // Constructor with arguments:
  ImageBlock(ImageChunk &chunk, int block_x_size)
      : block_x_index((int)(chunk.getChunkActualRowIndex() / block_x_size)), block_x_size(block_x_size), 
      block_y_size(chunk.y_size), block_y_index(chunk.y_index) {
    feature_count = chunk.getSize();
    chunks[chunk.getChunkActualRowIndex()] = chunk.getChunk();
    key = block_x_index * 10000 + block_y_index;
}

  std::map<int, std::vector<float>> &getBlock() { return chunks; }

  long &getKey() { return key; }

  ImageBlock &getValue() { return *this; }

  int getActualBlockSize() { return chunks.size(); }

  int getActualBlockChunksSize() { return chunks.begin()->second.size();}

  void merge(ImageBlock &addMeIn) {
    std::map<int, std::vector<float>> &rhs = addMeIn.getBlock();
    auto iter = rhs.begin();
    while (iter != rhs.end()) {
      int myKey = (*iter).first;
      if (chunks.count(myKey) == 0) {
        chunks[myKey] = (*iter).second;
        //std::cout << "[ImageBlock] Merging bucket " << key << " adding key "
                  //<< myKey << std::endl;
      } else {
        //std::cout << "[ImageBlock] Merging bucket " << key << " NOT adding key "
                  //<< myKey << std::endl;
      }
      ++iter;
    }
  }
};

class ImageMatrix {

public:
  int num_row;
  int num_col;
  float* data;

  // Default constructor:
  ImageMatrix() {}

  // Default destructor:
  ~ImageMatrix() {}

  // Constructor with arguments:
  ImageMatrix(int num_row, int num_col)
      : num_row(num_row), num_col(num_col) {}


  int getRowSize() { return num_row; }

  int getColSize() { return num_col; }

  float* getData() { return data; }

};

std::vector<Image> loadRandomImages(int width, int height, int channels, int numOfImages) {
    // Seed the random number generator
    std::random_device rd;  
    // Initialize the Mersenne Twister engine
    std::mt19937 gen(rd());
    // Define the range
    std::uniform_real_distribution<float> distribution(0, 1);

    // Vector to store images
    std::vector<Image> images;
    int imageSize = width * height;
    for (int imageCount = 0; imageCount < numOfImages; imageCount++) {
        // Create a new Image object
        Image newImage(imageCount, width, height, channels);

        for (int c = 0; c < channels; c++) {
            float* imageData = new float[imageSize];
            // Generate random data for the channel
            for (int i = 0; i < width * height; i++) {
                // Generate random data based on gen()
                float data = distribution(gen);
                // Store the data in the image data array
                imageData[i] = data;
            }
            // Add channel to the image
            newImage.setMatrixAtIndex(c, imageData);
        }

        // Add the constructed image to the vector
        images.push_back(newImage);

    }

    return images;
}

std::vector<ImageChunk> img_to_chunks(Image& image, int block_y, int strides, int kernel, int padding) {
    std::vector<ImageChunk> result;

    // int num_y_blocks = ceil(image.get_conv2d_matrix_cols(kernel, strides, padding) / static_cast<float>(block_y));//2*3*3 /32 = 1
    int windows = image.get_conv2d_window_count(kernel, strides, padding); //4*4 = 16, each chunk denotes each window
    // result.resize(windows);

    // std::cout << "[img_to_chunks] windows: " << windows << ", kernel: " << kernel << ", strides: " << strides << ", padding: " << padding << std::endl;

    int channels = image.get_num_channels();//2
    int row_start = image.getKey() * windows; // 0*16, 1*16, 2*16, for each image
    int counter = 0;
    std::shared_ptr<ImageChunk> chunk = nullptr;

    for (int w = 0; w < windows; w++) {
        counter = 0;
        chunk = std::make_shared<ImageChunk>(block_y, block_y, w, counter, row_start);//32,32,0,0,0 for first window

        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < kernel; i++) {
                for (int j = 0; j < kernel; j++) {

                    if (chunk->getSize() > 0 && chunk->getSize() % block_y == 0) {
                        // std::cout << chunk->getSize() << std::endl;
                        result.push_back(*chunk);
                        counter++;
                        chunk = std::make_shared<ImageChunk>(block_y, block_y, w, counter, row_start);
                    }

                    chunk->getChunk().push_back(image.get_win_value(w, i, j, strides, c, kernel, padding));
                }
            }
        }

        if (chunk != nullptr) {
            if (chunk->getSize() % block_y == 0) {
                result.push_back(*chunk);
                counter++;
                chunk = std::make_shared<ImageChunk>(block_y, block_y, w, counter, row_start);
            }

            chunk->getChunk().push_back(1);
            result.push_back(*chunk);
        }
    }

    return result;
}


std::vector<ImageChunk> kernel_to_chunks(Image& kernel, int block_y) {
    std::vector<ImageChunk> result;

    int y_index = 0;

    int row_width = kernel.x * kernel.y * kernel.z;
    int block_row = 0;
    int block_row_start = kernel.getKey();
    int channels = kernel.get_num_channels();
    int size = kernel.x * kernel.y;
    std::shared_ptr<ImageChunk> chunk = std::make_shared<ImageChunk>(block_y, block_y, block_row, y_index, block_row_start);

    for (int c = 0; c < channels; c++) {
        float* data = kernel.getMatrixAtIndex(c);

        for (int i = 0; i < size; i++) {
          if (chunk->getSize() > 0 && chunk->getSize() % block_y == 0) {
            result.push_back(*chunk);
            y_index++;
            chunk = std::make_shared<ImageChunk>(block_y, block_y, block_row, y_index, block_row_start);
          }
          chunk->getChunk().push_back(data[i]);
        }
      }

      // Add the last chunk of last row
      // Set placeholder for bias
      if (chunk->getSize() % block_y == 0) {
        result.push_back(*chunk);
        y_index++;
        chunk = std::make_shared<ImageChunk>(block_y, block_y, block_row, y_index, block_row_start);
      }

      chunk->getChunk().push_back(0);
      result.push_back(*chunk);

      return result;
}

std::vector<ImageBlock*> chunks_to_blocks(std::vector<ImageChunk>& chunks, int block_x) {
    std::vector<ImageBlock*> blocks;
    blocks.reserve(chunks.size());
    std::map<long, std::vector<ImageBlock*>> blockMap;
    std::vector<ImageBlock*> mergedBlocks;

    // Group chunks by their actual row index
    for (auto& chunk : chunks) {
      ImageBlock* block = new ImageBlock(chunk, block_x);
      blocks.push_back(block);
    }
    for (auto& block: blocks){
      blockMap[block->getKey()].push_back(block);
    }
    mergedBlocks.reserve(blockMap.size());
    for (auto& entry: blockMap){
      ImageBlock* mergedBlock = entry.second[0];
      for (size_t i = 1; i < entry.second.size(); ++i) {
        mergedBlock->merge(*entry.second[i]);
      }
      mergedBlocks.push_back(mergedBlock);
    }


    return mergedBlocks;
}

std::vector<ImageBlock*> kernelchunks_to_blocks(std::vector<ImageChunk>& chunks, int block_x) {
    std::vector<ImageBlock*> blocks;
    blocks.reserve(chunks.size());
    std::map<long, std::vector<ImageBlock*>> blockMap;
    std::vector<ImageBlock*> mergedBlocks;
    // Group chunks by their actual row index
    for (auto& chunk : chunks) {
      ImageBlock* block = new ImageBlock(chunk, block_x);
      blocks.push_back(block);
    }
    for (auto& block: blocks){
      blockMap[block->getKey()].push_back(block);
    }

    for (auto& entry: blockMap){
      ImageBlock* mergedBlock = entry.second[0];
      for (size_t i = 1; i < entry.second.size(); ++i) {
        mergedBlock->merge(*entry.second[i]);
      }
      mergedBlocks.push_back(mergedBlock);
    }



    return mergedBlocks;
}

ImageMatrix blocks_to_matrix(std::vector<ImageBlock*>& blocks, int block_x, int block_y, bool padding, int total_features, int total_rows) {

    int total_cols = total_features; // Assuming total_features represents the number of columns
    float* data = new float[total_rows * total_cols]; // Allocate memory for the data array

    for (auto& block_ptr : blocks) {
      ImageBlock& block = *block_ptr;
        int real_block_x = padding ? block_x : block.getActualBlockSize();
        int real_block_y = padding ? block_y : std::min(block_y, total_features - block.block_y_index * block_y);

        const auto& chunk = block.getBlock();
        int offset = block.block_x_index * block.block_x_size;
        int offset_y = block.block_y_index * block.block_y_size;

        // Copy data from the block into the data array
        for (int x = 0; x < real_block_x; x++) {
            for (int y = 0; y < real_block_y; y++) {
                float value = (x >= block.getActualBlockSize() || y >= block.feature_count) ? 0 : chunk.at(offset + x).at(y);
                data[x * total_cols + y + offset_y] = value;
            }
        }
    }

    ImageMatrix myData(total_rows, total_cols); // Create an ImageMatrix object
    myData.data = data; // Assign the data array to the ImageMatrix object

    return myData;

}

ImageMatrix kernelblocks_to_matrix(std::vector<ImageBlock*>& blocks, int block_x, int block_y, bool padding, int total_features, int total_rows) {

    int total_cols = total_features; // Assuming total_features represents the number of columns
    float* data = new float[total_rows * total_cols]; // Allocate memory for the data array



    for (auto& block_ptr : blocks) {
      ImageBlock& block = *block_ptr;
        int real_block_x = padding ? block_x : block.getActualBlockSize();
        int real_block_y = padding ? block_y : std::min(block_y, total_rows - block.block_y_index * block_y);

        const auto& chunk = block.getBlock();
        int offset = block.block_x_index * block.block_x_size;
        int offset_y = block.block_y_index * block.block_y_size;
        // Copy data from the block into the data array
        for (int x = 0; x < real_block_x; x++) {
            for (int y = 0; y < real_block_y; y++) {
                float value = (x >= block.getActualBlockSize() || y >= block.feature_count) ? 0 : chunk.at(offset + x).at(y);
                data[(y + offset_y) * total_cols + x] = value;
            }
        }
    }

    ImageMatrix myData(total_rows, total_cols); // Create an ImageMatrix object
    myData.data = data; // Assign the data array to the ImageMatrix object

    return myData;

}

ImageMatrix copy_bias_to_matrix(ImageMatrix kernelMatrix, float* bias){
    int lastRow = kernelMatrix.getRowSize() - 1;
    int numCols = kernelMatrix.getColSize();
    for (int i = 0; i < numCols; ++i) {
        kernelMatrix.data[lastRow * numCols + i] = bias[i];
    }
    return kernelMatrix;
}

float* loadRandomBias(int size) {
    // Create a random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f); // Range of random float values

    // Allocate memory for the array
    float* arr = new float[size];

    // Fill the array with random float values
    for (int i = 0; i < size; ++i) {
        arr[i] = dis(gen);
    }

    return arr;
}


ImageMatrix convertImage(std::vector<Image>& images, int window_items, int strides, int kWidth, int kChannels, int padding, int block_x, int block_y) {

    int finalRowSize = images.size() * images[0].get_conv2d_window_count(kWidth, strides, padding);
    int finalColSize = kChannels*kWidth*kWidth+1;
    std::vector<ImageChunk> chunks;
    for (auto& image : images){
      std::vector<ImageChunk> newChunks = img_to_chunks(image, window_items, strides, kWidth, padding);
      chunks.insert(chunks.end(), newChunks.begin(), newChunks.end());
    }

    auto blocks = chunks_to_blocks(chunks, block_x);
    auto matrix = blocks_to_matrix(blocks, block_x, block_y, padding, finalColSize, finalRowSize);// 2*3*3+1=19, kChannels*kHeight*kWidth+1=19; get_conv2d_window_count() = 16
    return matrix;
}

ImageMatrix convertKernel(std::vector<Image>& kernels, int window_items, int kWidth, int kChannels, bool block_padding, int block_x, int block_y, float* bias) {
    int finalRowSize = kChannels*kWidth*kWidth+1;
    int finalColSize = kernels.size();
    // float* bias = loadRandomBias(kernels.size());
    std::vector<ImageChunk> chunks;
    for (auto& kernel : kernels){
      std::vector<ImageChunk> newChunks = kernel_to_chunks(kernel, window_items);
      chunks.insert(chunks.end(), newChunks.begin(), newChunks.end());
    }

    auto kernelBlocks = kernelchunks_to_blocks(chunks, block_x);
    auto kernelMatrix = kernelblocks_to_matrix(kernelBlocks, block_x, block_y, block_padding, finalColSize, finalRowSize);
    auto kernelBiasMatrix = copy_bias_to_matrix(kernelMatrix, bias);

    return kernelBiasMatrix;
}

std::vector<Image> loadFormData(int width, int height, int channels, int numOfImages, float* values) {

    // Vector to store images
    std::vector<Image> images;
    int imageSize = width * height;
    for (int imageCount = 0; imageCount < numOfImages; imageCount++) {
        // Create a new Image object
        Image newImage(imageCount, width, height, channels);

        for (int c = 0; c < channels; c++) {
            float* imageData = new float[imageSize];
            // Generate random data for the channel
            for (int i = 0; i < width * height; i++) {
                // Store the data in the image data array
                imageData[i] = values[c * imageSize + i];
            }
            // Add channel to the image
            newImage.setMatrixAtIndex(c, imageData);
        }

        // Add the constructed image to the vector
        images.push_back(newImage);
    }

    return images;
}

ImageMatrix convert2Matrix(int cnn_filter_dims[], int numOfImages, float* values, float* bias) {
  
    std::vector<Image> kernels = loadFormData(cnn_filter_dims[0], cnn_filter_dims[1], cnn_filter_dims[2], numOfImages, values);
    ImageMatrix kernelData = convertKernel(kernels, cnn_filter_dims[0]*cnn_filter_dims[1], cnn_filter_dims[0], cnn_filter_dims[2], false, 32, cnn_filter_dims[0]*cnn_filter_dims[1], bias);
    return kernelData;
}

ImageMatrix convert2Matrix(int cnn_filter_dims[], int input_dims[], int numOfImages, float* values) {

    std::vector<Image> images = loadRandomImages(input_dims[0], input_dims[1], input_dims[2], numOfImages);
    // std::vector<Image> images = loadFormData(input_dims[0], input_dims[1], input_dims[2], numOfImages, values);
    ImageMatrix imageData = convertImage(images, cnn_filter_dims[0]*cnn_filter_dims[1], 1, cnn_filter_dims[0], cnn_filter_dims[2], false, 32, cnn_filter_dims[0]*cnn_filter_dims[1]);
    return imageData;
}

std::vector<std::vector<float>> chunks2Blocks(std::vector<ImageChunk>& chunks, int block_x_width, int block_y_width) {
    std::vector<std::vector<float>> blocks;
    int blocks_row_size = (chunks.size() + block_x_width -1) / block_x_width;//392
    // int single_chunk_size = chunks[0].getSize();
    // int total_chunk_size = (block_y_width - 1) * single_chunk_size + 1;
    int total_chunk_size = chunks[0].getSize();
    blocks.resize(blocks_row_size);
    for (auto& block : blocks) {
        block.resize(block_x_width * total_chunk_size);
    }
    // for (int i = 0; i < blocks_row_size; ++i) {
    //     blocks[i] = new float[block_x_width * total_chunk_size];
    // }

    for (auto& chunk : chunks) {
        // Calculate block index for the chunk
        int block_index = (chunk.block_row + chunk.block_row_start) / block_x_width;


        // Calculate starting position in the block
        int start_row = ((chunk.block_row + chunk.block_row_start) % block_x_width) * total_chunk_size;
        // int start_col = chunk.y_index * single_chunk_size;

        // Copy values to the block
        std::copy(chunk.chunk.begin(), chunk.chunk.end(), blocks[block_index].begin() + start_row);
        // std::copy(chunk.chunk.begin(), chunk.chunk.end(), 
        //           blocks[block_index] + start_row);
    }
    return blocks;

}

std::vector<std::vector<float>> convert2Blocks(int width, int height, int channels, int numOfImages, int kWidth, int kChannels, int strides, int padding, int block_x_width) {
  auto images = loadRandomImages(width, height, channels, numOfImages);

  int finalRowSize = images.size() * images[0].get_conv2d_window_count(kWidth, strides, padding);

  int finalColSize = kChannels*kWidth*kWidth+1;

  std::vector<ImageChunk> chunks;
  for (auto& image : images){
    std::vector<ImageChunk> newChunks = img_to_chunks(image, finalColSize, strides, kWidth, padding);
    chunks.insert(chunks.end(), newChunks.begin(), newChunks.end());
  }


  auto blocks = chunks2Blocks(chunks, block_x_width, finalColSize);
  return blocks;
}

std::vector<std::vector<float>> convert2Matrix(int width, int height, int channels, int numOfImages, int kWidth, int kChannels, int strides, int padding, int block_x_width) {
  auto images = loadRandomImages(width, height, channels, numOfImages);
  int finalRowSize = images.size() * images[0].get_conv2d_window_count(kWidth, strides, padding);
  int finalColSize = kChannels*kWidth*kWidth+1;
  std::vector<ImageChunk> chunks;
  for (auto& image : images){
    std::vector<ImageChunk> newChunks = img_to_chunks(image, finalColSize, strides, kWidth, padding);
    chunks.insert(chunks.end(), newChunks.begin(), newChunks.end());
  }

  auto blocks = chunks2Blocks(chunks, block_x_width, finalColSize);
  return blocks;
}

// ImageMatrix convert2Matrix(int width, int height, int channels, int numOfImages, float* values, bool isKernel, int kernelSize, float* bias) {
//   if (isKernel) {
//     std::vector<Image> kernels = loadFormData(width, height, channels, numOfImages, values);
//     ImageMatrix kernelData = convertKernel(kernels, kernelSize, width, channels, false, 32, kernelSize, bias);
//     return kernelData;
//   }
//   else {
//     std::vector<Image> images = loadFormData(width, height, channels, numOfImages, values);
//     ImageMatrix imageData = convertImage(images, kernelSize, 1, width, channels, false, 32, kernelSize);
//     return imageData;
//   }

// }


// int main(int argc, char** argv) {

//     int height = 6, width = 6, channels = 2, numOfImages = 3;
//     int kHeight = 3, kWidth = 3, kChannels = 2, numOfFilters = 3;
//     std::cout << "Loading images....." << std::endl;
//     auto images = loadRandomImages(width, height, channels, numOfImages);
//     std::cout << "Loading kernel....." << std::endl;
//     auto kernels = loadRandomImages(kHeight, kWidth, kChannels, numOfFilters);
//     std::cout << "Loading bias data..." << std::endl;
//     float* bias = loadRandomBias(numOfFilters);

//     int num_items_in_window = kHeight*kWidth; //3*3=9
//     int strides = 1; //step size
//     int padding = 0; // 0 denote no padding; 1 denote padding in edge; Used in Image part
//     bool block_padding = false; // Used in kernel part
//     int block_x = 32; // the row size of virtual block, the block is build on several chunks, usually one channel construct one block
//     int block_y = num_items_in_window; // the col size of virtual block



//     ImageMatrix imageData = convertImage(images, num_items_in_window, strides, kWidth, kChannels, padding, block_x, block_y);
//     ImageMatrix kernelData = convertKernel(kernels, num_items_in_window, kWidth, kChannels, block_padding, block_x, block_y);


//     // auto kernelData = convertKernel(kernels[0]);

//     // Image image = images[0];
//     // auto chunks = img_to_chunks(image, window_items, strides, kWidth, padding);
//     // auto blocks = chunks_to_blocks(chunks, block_x);
//     // auto matrix = blocks_to_matrix(blocks, block_x, block_y, padding, 19, 16);// 2*3*3+1=19, kChannels*kHeight*kWidth+1=19; get_conv2d_window_count() = 16


//     // auto kernelChunks = kernel_to_chunks(kernels[0], kHeight*kWidth);
//     // auto kernelBlocks = kernelchunks_to_blocks(kernelChunks, block_x);
//     // auto kernelMatrix = kernelblocks_to_matrix(kernelBlocks, block_x, block_y, block_padding, 1, 19);
//     // auto kernelBiasMatrix = copy_bias_to_matrix(kernelMatrix, bias);

//     std::cout << "Finished converting" << std::endl;

// }
