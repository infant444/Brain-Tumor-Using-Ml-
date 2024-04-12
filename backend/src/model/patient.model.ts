import { Types,Schema,model } from "mongoose";



export interface Tumor_Info{
    segmantation:string;
    classify:string;
    size:number;
    confidence_level:number;
  }
  
export interface Patient{
    name:string;
    dob:string;
    age:string;
    gender:string;
    symptoms:string[];
    given_img:string;
    predicted_img:string;
    brain_tumor:boolean;
    tumor_imfo:Tumor_Info;
    user_id:Types.ObjectId;

}
  
export const TumorinfoSchema=new Schema<Tumor_Info>({
    segmantation:{type:String},
    classify:{type:String,required:true},
    size:{type:Number},
    confidence_level:{type:Number},
});

export const Patient_Schame=new Schema<Patient>({
    name:{type:String,required:true},
    dob:{type:String,required:true},
    age:{type:String,required:true},
    gender:{type:String,required:true},
    symptoms:{type:[String]},
    given_img:{type:String,required:true},
    predicted_img:{type:String},
    brain_tumor:{type:Boolean,default:false},
    tumor_imfo:{type:TumorinfoSchema,required:true},
    user_id:{type:Schema.Types.ObjectId,required:true}
},
{
    timestamps: true,
    toJSON:{
        virtuals: true
    },
    toObject:{
        virtuals: true
    }
}
);

export const PatientModel=model<Patient>('Patient_info',Patient_Schame);