// Shared pixel-level helpers used by several module widgets.

export function clamp255(v) {
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// Multiply brightness and add sensor-style grain (stronger in shadows).
export function applyExposureAndNoise(ctx, w, h, brightness, noiseAmount) {
  if (brightness === 1 && noiseAmount <= 0) return;
  const imageData = ctx.getImageData(0, 0, w, h);
  const data = imageData.data;
  const noiseStrength = noiseAmount * 55;

  for (let i = 0; i < data.length; i += 4) {
    let r = data[i] * brightness;
    let g = data[i + 1] * brightness;
    let b = data[i + 2] * brightness;

    if (noiseStrength > 0.5) {
      const luminance = (r + g + b) / 3 / 255;
      const shadowBoost = 1.6 - luminance;
      const n = (Math.random() - 0.5) * noiseStrength * shadowBoost;
      r += n;
      g += n * 0.9;
      b += n * 1.1;
    }

    data[i] = clamp255(r);
    data[i + 1] = clamp255(g);
    data[i + 2] = clamp255(b);
  }

  ctx.putImageData(imageData, 0, 0);
}

// Color temperature (Kelvin) -> RGB, Tanner Helland approximation.
export function kelvinToRGB(kelvin) {
  const t = kelvin / 100;
  let r, g, b;

  if (t <= 66) {
    r = 255;
    g = clamp255(99.4708025861 * Math.log(t) - 161.1195681661);
    b = t <= 19 ? 0 : clamp255(138.5177312231 * Math.log(t - 10) - 305.0447927307);
  } else {
    r = clamp255(329.698727446 * Math.pow(t - 60, -0.1332047592));
    g = clamp255(288.1221695283 * Math.pow(t - 60, -0.0755148492));
    b = 255;
  }
  return [r, g, b];
}

// 64-bucket luminance histogram from canvas pixels (samples every 4th pixel for speed).
export function computeHistogram(ctx, w, h, buckets = 64) {
  const data = ctx.getImageData(0, 0, w, h).data;
  const hist = new Array(buckets).fill(0);
  for (let i = 0; i < data.length; i += 16) {
    const lum = 0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2];
    hist[Math.min(buckets - 1, Math.floor((lum / 256) * buckets))]++;
  }
  return hist;
}

export function roundRect(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}
