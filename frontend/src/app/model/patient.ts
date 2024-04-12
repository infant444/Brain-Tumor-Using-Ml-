import { Tumor_Info } from "./brain";

export class Patient{
  id!:string;
  name!:string;
  dob!:string;
  age!:string;
  gender!:string;
  symptoms?:string[];
  given_img!:string;
  predicted_img!:string;
  brain_tumor!:boolean;
  tumor_imfo!:Tumor_Info;
}
