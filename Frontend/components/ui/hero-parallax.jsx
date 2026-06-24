"use client";
import React from "react";
import { motion, useScroll, useTransform, useSpring } from "framer-motion";
import Link from "next/link";
import { useRouter } from "next/navigation";

export const HeroParallax = ({
  products
}) => {
  const firstRow = products.slice(0, 5);
  const secondRow = products.slice(5, 10);
  const thirdRow = products.slice(10, 15);
  const ref = React.useRef(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });

  const springConfig = { stiffness: 300, damping: 30, bounce: 100 };

  const translateX = useSpring(useTransform(scrollYProgress, [0, 1], [0, 1000]), springConfig);
  const translateXReverse = useSpring(useTransform(scrollYProgress, [0, 1], [0, -1000]), springConfig);
  const rotateX = useSpring(useTransform(scrollYProgress, [0, 0.2], [15, 0]), springConfig);
  const opacity = useSpring(useTransform(scrollYProgress, [0, 0.2], [0.2, 1]), springConfig);
  const rotateZ = useSpring(useTransform(scrollYProgress, [0, 0.2], [20, 0]), springConfig);
  const translateY = useSpring(useTransform(scrollYProgress, [0, 0.2], [-700, 500]), springConfig);
  return (
    (<div
      ref={ref}
      className="h-[300vh] py-40 overflow-hidden  antialiased relative flex flex-col self-auto [perspective:1000px] [transform-style:preserve-3d]">
      <Header />
      <motion.div
        style={{
          rotateX,
          rotateZ,
          translateY,
          opacity,
        }}
        className="">
        <motion.div className="flex flex-row-reverse space-x-reverse space-x-20 mb-20">
          {firstRow.map((product) => (
            <ProductCard product={product} translate={translateX} key={product.title} />
          ))}
        </motion.div>
        <motion.div className="flex flex-row  mb-20 space-x-20 ">
          {secondRow.map((product) => (
            <ProductCard product={product} translate={translateXReverse} key={product.title} />
          ))}
        </motion.div>
        <motion.div className="flex flex-row-reverse space-x-reverse space-x-20">
          {thirdRow.map((product) => (
            <ProductCard product={product} translate={translateX} key={product.title} />
          ))}
        </motion.div>
      </motion.div>
    </div>)
  );
};

export const Header = () => {
  const router = useRouter();

  const handleNavigate = () => {
    router.push('/dashboard');
  };

  return (
    <div className="max-w-7xl relative mx-auto py-20 md:py-40 px-4 w-full left-0 top-0">
      <h1 className="text-2xl md:text-7xl font-bold dark:text-white">
        The Ultimate <br /> 
        <span className="bg-clip-text text-transparent bg-gradient-to-r from-blue-500 to-purple-500">
          Power Load Predictor
        </span>
      </h1>
      <p className="max-w-2xl text-base md:text-xl mt-8 text-white">
        Advanced AI-powered solution for predicting and analyzing power load variations in Delhi NCT. 
        Our platform integrates weather patterns, seasonal changes, and urban development factors to 
        deliver precise load forecasting and optimization recommendations.
      </p>
      <div className="mt-8 flex gap-4">
        <button 
          onClick={handleNavigate}
          className="px-6 py-3 rounded-full bg-blue-500 hover:bg-blue-600 text-white font-medium transition-colors"
        >
          Start Predicting
        </button>
        <button 
          onClick={handleNavigate}
          className="px-6 py-3 rounded-full bg-neutral-800 hover:bg-neutral-700 text-white font-medium transition-colors"
        >
          View Insights
        </button>
      </div>
    </div>
  );
};

export const ProductCard = ({
  product,
  translate
}) => {
  const router = useRouter();

  const handleCardClick = () => {
    router.push('/dashboard');
  };

  return (
    <motion.div
      style={{
        x: translate,
      }}
      whileHover={{
        y: -20,
      }}
      onClick={handleCardClick}
      className="group/product h-96 w-[30rem] relative flex-shrink-0 cursor-pointer"
    >
      <div 
        className="block group-hover/product:shadow-2xl h-full w-full bg-gradient-to-br from-neutral-900 to-neutral-800 rounded-xl p-8 transition-all duration-300"
      >
        <div className="flex flex-col h-full">
          <div className="absolute -top-8 left-1/2 -translate-x-1/2 bg-gradient-to-br from-blue-600 to-purple-600 p-4 rounded-2xl shadow-xl">
            {React.createElement(product.icon, {
              size: 32,
              className: "text-white"
            })}
          </div>
          <div className="mt-8">
            <h3 className="text-2xl font-bold text-white text-center mb-3">{product.title}</h3>
            <p className="text-neutral-300 text-sm mb-4 text-center">{product.description}</p>
            
            <div className="space-y-2 mb-6">
              {product.features?.map((feature, index) => (
                <div key={index} className="flex items-center text-sm text-neutral-400">
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mr-2" />
                  {feature}
                </div>
              ))}
            </div>

            <div className="mt-auto">
              <div className="flex flex-wrap gap-2 justify-center">
                {product.tags.map((tag, index) => (
                  <span 
                    key={index}
                    className="px-3 py-1 text-xs rounded-full bg-blue-500/10 text-blue-400"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </motion.div>
  );
};