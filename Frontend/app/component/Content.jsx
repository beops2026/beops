"use client";
import { GlobeDemo } from "./Hero";
import { HeroParallax } from "@/components/ui/hero-parallax";
import { products } from "../../data/product";

function Content() {
  return (
    <div className="w-full bg-black">
      <div className="h-screen flex items-center justify-center p-5">
        <GlobeDemo />
      </div>
      <HeroParallax products={products} />
    </div>
  );
}

export default Content;