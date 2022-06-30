function [X y Xt yt] = transformMNIST()
    [imgs labels] = readMNIST(
        'train-images-idx3-ubyte', 
        'train-labels-idx1-ubyte',
        60000,
        0);
    [X y] = transform(imgs, labels);
    [imgs labels] = readMNIST(
        't10k-images-idx3-ubyte',
        't10k-labels-idx1-ubyte',
        10000,
        0);
    [Xt yt] = transform(imgs, labels);
end

function [oimgs labels] = transform(imgs, labels)
    labels(labels==0) = 10;
    [h w iSize] = size(imgs);
    oimgs = zeros(iSize, h * w);

    for i = 1:iSize
        oimgs(i,:) = reshape(imgs(:,:,i), 1, 400);
    end
end

